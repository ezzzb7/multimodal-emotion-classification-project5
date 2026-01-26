"""
重新训练 HP1 最佳配置并保存模型
然后基于此模型做 Bad Case 改进实验
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, DistilBertTokenizer

sys.path.insert(0, os.path.dirname(__file__))

from run_experiment_optimized import OptimizedMultimodalClassifier, OPTIMIZED_CONFIG
from data.data_loader import get_data_loaders
from utils.train_utils import set_seed, compute_metrics, EarlyStopping


# ========== HP1 最佳配置 ==========
HP1_CONFIG = {
    **OPTIMIZED_CONFIG,
    'dropout': 0.2,
    'lr_classifier': 1e-4,
    'weight_decay': 0.01,
}


# ========== Focal Loss ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ========== Label Smoothing ==========
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        smooth_label = one_hot * confidence + (1 - one_hot) * smooth_value
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(smooth_label * log_prob).sum(dim=1)
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def train_model(exp_id, exp_name, config, loss_type='ce', class_weights=None, 
                focal_gamma=2.0, label_smoothing=0.0, num_epochs=20):
    """训练模型"""
    
    print(f"\n{'='*60}")
    print(f"[{exp_id}] {exp_name}")
    print(f"{'='*60}")
    
    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 加载 Tokenizer
    print("加载 Tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=True)
    
    # 使用与 HP1 完全相同的数据加载方式
    print("加载数据...")
    train_loader, val_loader, _ = get_data_loaders(
        data_dir=config['data_dir'],
        train_label_file=config['train_label'],
        batch_size=config['batch_size'],
        val_ratio=config['val_ratio'],
        num_workers=0,
        seed=config['seed'],
        force_resplit=True  # 重要：每次重新划分确保一致
    )
    print(f"  训练集: {len(train_loader.dataset)}, 验证集: {len(val_loader.dataset)}")
    
    # 创建模型
    print("创建模型...")
    model = OptimizedMultimodalClassifier(
        num_classes=3,
        feature_dim=config['feature_dim'],
        fusion_type='cross_attention',
        dropout=config['dropout'],
        unfreeze_text_layers=config['unfreeze_text_layers'],
        unfreeze_image_layers=config['unfreeze_image_layers']
    ).to(device)
    
    # 损失函数
    if class_weights is None:
        class_weights = [1.0, 1.5, 3.0]
    weight_tensor = torch.FloatTensor(class_weights).to(device)
    
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=weight_tensor, gamma=focal_gamma)
        print(f"  损失函数: Focal Loss (gamma={focal_gamma})")
    elif loss_type == 'label_smoothing':
        criterion = LabelSmoothingLoss(num_classes=3, smoothing=label_smoothing, weight=weight_tensor)
        print(f"  损失函数: Label Smoothing ({label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print(f"  损失函数: CrossEntropy, 类别权重: {class_weights}")
    
    # 优化器
    param_groups = model.get_param_groups(
        lr_pretrained=config['lr_pretrained'],
        lr_fusion=config['lr_fusion'],
        lr_classifier=config['lr_classifier']
    )
    optimizer = AdamW(param_groups, weight_decay=config['weight_decay'])
    
    total_steps = len(train_loader) * num_epochs // config['accumulation_steps']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    early_stopping = EarlyStopping(patience=7, mode='max')
    
    # 训练
    print(f"\n开始训练 ({num_epochs} epochs)...")
    sys.stdout.flush()
    
    best_val_acc = 0
    best_val_f1 = 0
    best_epoch = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        print(f"  Epoch {epoch+1}/{num_epochs} [", end="", flush=True)
        
        for step, batch in enumerate(train_loader):
            texts = batch['text']
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels) / config['accumulation_steps']
            loss.backward()
            
            if (step + 1) % config['accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if (step + 1) % 40 == 0:
                print(".", end="", flush=True)
        
        print("] ", end="", flush=True)
        
        # 验证
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                texts = batch['text']
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                logits = model(input_ids, attention_mask, images)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_metrics = compute_metrics(val_preds, val_labels)
        print(f"Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}", flush=True)
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    ✓ 新最佳!", flush=True)
        
        if early_stopping(val_metrics['accuracy'], epoch):
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # 保存模型
    if best_model_state:
        os.makedirs('experiments/checkpoints', exist_ok=True)
        save_path = f'experiments/checkpoints/{exp_id}_best.pth'
        torch.save(best_model_state, save_path)
        print(f"\n✓ 模型已保存: {save_path}")
    
    print(f"  最佳 Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    return {
        'exp_id': exp_id,
        'exp_name': exp_name,
        'val_acc': best_val_acc,
        'val_f1': best_val_f1,
        'best_epoch': best_epoch
    }


def run_hp1_and_improvements():
    """先训练 HP1，然后进行改进实验"""
    
    print("\n" + "="*70)
    print("HP1 最佳配置训练 + Bad Case 改进实验")
    print("="*70)
    
    # 断点恢复
    checkpoint_file = 'experiments/hp1_improvement_checkpoint.json'
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        completed = set(checkpoint.get('completed', []))
        results = checkpoint.get('results', [])
        print(f"✓ 从断点恢复，已完成: {completed}")
    else:
        completed = set()
        results = []
    
    experiments = [
        # (exp_id, exp_name, loss_type, class_weights, focal_gamma, label_smoothing)
        ('HP1_BEST', 'HP1最佳配置(基线)', 'ce', [1.0, 1.5, 3.0], 2.0, 0.0),
        ('HP1_W2', '增加neg权重(1,2,3)', 'ce', [1.0, 2.0, 3.0], 2.0, 0.0),
        ('HP1_W3', '增加neg权重(1,2.5,3.5)', 'ce', [1.0, 2.5, 3.5], 2.0, 0.0),
        ('HP1_FOCAL', 'Focal Loss', 'focal', [1.0, 1.5, 3.0], 2.0, 0.0),
        ('HP1_FOCAL_W', 'Focal+调整权重', 'focal', [1.0, 2.0, 3.0], 2.0, 0.0),
        ('HP1_LS', 'Label Smoothing 0.1', 'label_smoothing', [1.0, 1.5, 3.0], 2.0, 0.1),
    ]
    
    for exp_id, exp_name, loss_type, weights, gamma, smooth in experiments:
        if exp_id in completed:
            print(f"\n[{exp_id}] 已完成，跳过")
            continue
        
        try:
            result = train_model(
                exp_id=exp_id,
                exp_name=exp_name,
                config=HP1_CONFIG,
                loss_type=loss_type,
                class_weights=weights,
                focal_gamma=gamma,
                label_smoothing=smooth,
                num_epochs=20
            )
            results.append(result)
            completed.add(exp_id)
            
            # 保存断点
            with open(checkpoint_file, 'w') as f:
                json.dump({'completed': list(completed), 'results': results}, f, indent=2)
            print(f"  ✓ 断点已保存")
            
        except Exception as e:
            print(f"  ❌ 实验失败: {e}")
            import traceback
            traceback.print_exc()
            with open(checkpoint_file, 'w') as f:
                json.dump({'completed': list(completed), 'results': results}, f, indent=2)
            raise
    
    # 打印结果
    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)
    print(f"{'ID':<15} {'策略':<25} {'Val Acc':<10} {'Val F1':<10}")
    print("-"*60)
    
    for r in sorted(results, key=lambda x: -x['val_acc']):
        print(f"{r['exp_id']:<15} {r['exp_name']:<25} {r['val_acc']:.4f}     {r['val_f1']:.4f}")
    
    if results:
        best = max(results, key=lambda x: x['val_acc'])
        print(f"\n🏆 最佳结果: {best['exp_name']}")
        print(f"   Val Acc: {best['val_acc']:.4f}")
        print(f"   模型: experiments/checkpoints/{best['exp_id']}_best.pth")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='运行实验')
    parser.add_argument('--hp1-only', action='store_true', help='只训练HP1基线')
    
    args = parser.parse_args()
    
    if args.hp1_only:
        # 只训练 HP1 基线
        result = train_model(
            exp_id='HP1_BEST',
            exp_name='HP1最佳配置',
            config=HP1_CONFIG,
            loss_type='ce',
            class_weights=[1.0, 1.5, 3.0],
            num_epochs=20
        )
        print(f"\nHP1 训练完成: Val Acc = {result['val_acc']:.4f}")
    elif args.run:
        run_hp1_and_improvements()
    else:
        print("使用方法:")
        print("  py -3.11 run_hp1_improvements.py --hp1-only  # 只训练HP1并保存模型")
        print("  py -3.11 run_hp1_improvements.py --run       # 训练HP1+改进实验")
