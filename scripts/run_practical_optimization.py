"""
基于数据驱动的务实优化方案

核心策略：
1. ✅ 不追求过于复杂的技巧（过采样/Mixup已验证无效）
2. ✅ 专注于提升当前最佳模型 (OPT_cross_attention 71.25%)
3. ✅ 通过超参数微调榨取最后的性能
4. ✅ 模型集成提升鲁棒性
5. ✅ 支持断点续传
"""
import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from run_experiment_optimized import OptimizedMultimodalClassifier, OPTIMIZED_CONFIG
from data.data_loader import get_data_loaders
from utils.train_utils import set_seed, compute_metrics, EarlyStopping


# ========== 断点续传文件 ==========
CHECKPOINT_FILE = 'experiments/hyperparam_checkpoint.json'
ENSEMBLE_CHECKPOINT_FILE = 'experiments/ensemble_checkpoint.json'


def load_checkpoint(checkpoint_file):
    """加载断点"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {'completed': [], 'results': []}


def save_checkpoint(checkpoint_file, data):
    """保存断点"""
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f, indent=2)


# ========== 超参数网格搜索 ==========
HYPERPARAM_GRID = {
    'dropout': [0.2, 0.3, 0.4],  # 当前0.3，尝试更小/更大
    'lr_classifier': [5e-5, 1e-4, 2e-4],  # 当前1e-4
    'weight_decay': [0.005, 0.01, 0.02],  # 当前0.01
}


def train_with_hyperparams(exp_id, dropout, lr_classifier, weight_decay):
    """使用指定超参数训练模型"""
    
    config = OPTIMIZED_CONFIG.copy()
    config['dropout'] = dropout
    config['lr_classifier'] = lr_classifier
    config['weight_decay'] = weight_decay
    
    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 离线加载tokenizer，避免网络问题
    print("加载 Tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=True)
    print("  ✓ Tokenizer 加载完成")
    
    # 加载数据
    print("加载数据...")
    train_loader, val_loader, _ = get_data_loaders(
        data_dir=config['data_dir'],
        train_label_file=config['train_label'],
        batch_size=config['batch_size'],
        val_ratio=config['val_ratio'],
        num_workers=0,
        seed=config['seed'],
        force_resplit=True
    )
    print(f"  ✓ 数据加载完成: 训练集 {len(train_loader.dataset)}, 验证集 {len(val_loader.dataset)}")
    
    # 创建模型
    print("创建模型...")
    model = OptimizedMultimodalClassifier(
        num_classes=3,
        feature_dim=config['feature_dim'],
        fusion_type='cross_attention',  # 使用最佳融合方法
        dropout=dropout,
        unfreeze_text_layers=config['unfreeze_text_layers'],
        unfreeze_image_layers=config['unfreeze_image_layers']
    ).to(device)
    print("  ✓ 模型创建完成")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {trainable_params:,} / {total_params:,}")
    
    # 训练设置
    print("设置优化器...")
    class_weights = torch.FloatTensor([1.0, 1.5, 3.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    param_groups = model.get_param_groups(
        lr_pretrained=config['lr_pretrained'],
        lr_fusion=config['lr_fusion'],
        lr_classifier=lr_classifier
    )
    optimizer = AdamW(param_groups, weight_decay=weight_decay)
    
    total_steps = len(train_loader) * config['num_epochs'] // config['accumulation_steps']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'], mode='max')
    print("  ✓ 优化器设置完成")
    
    # 快速训练（减少epoch）
    print("\n开始训练...")
    import sys
    sys.stdout.flush()
    
    best_val_acc = 0
    best_val_f1 = 0
    best_epoch = 0
    
    for epoch in range(15):  # 减少到15个epoch快速验证
        # 训练
        model.train()
        batch_count = 0
        print(f"  Epoch {epoch+1}/15 [", end="", flush=True)
        
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
            
            batch_count += 1
            # 每20个batch打印一个点 (共400个batch，显示20个点)
            if batch_count % 20 == 0:
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
            print(f"    ✓ 新最佳!", flush=True)
        
        if early_stopping(val_metrics['accuracy'], epoch):
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    return {
        'exp_id': exp_id,
        'dropout': dropout,
        'lr_classifier': lr_classifier,
        'weight_decay': weight_decay,
        'val_acc': best_val_acc,
        'val_f1': best_val_f1,
        'best_epoch': best_epoch
    }


def hyperparam_search():
    """超参数网格搜索（快速版，支持断点续传）"""
    
    print("\n" + "="*70)
    print("超参数网格搜索 (支持断点续传)")
    print("="*70)
    
    # 加载断点
    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    completed = set(checkpoint.get('completed', []))
    results = checkpoint.get('results', [])
    
    if completed:
        print(f"✓ 从断点恢复，已完成 {len(completed)} 个实验")
    
    # 只测试关键组合
    important_configs = [
        # (dropout, lr_classifier, weight_decay, exp_id)
        (0.2, 1e-4, 0.01, 'HP1'),    # 减少dropout
        (0.3, 1e-4, 0.02, 'HP2'),    # 增加正则化
        (0.3, 5e-5, 0.01, 'HP3'),    # 减小学习率
        (0.3, 2e-4, 0.01, 'HP4'),    # 增大学习率
        (0.25, 1e-4, 0.015, 'HP5'),  # 平衡配置
    ]
    
    for dropout, lr_clf, wd, exp_id in important_configs:
        # 跳过已完成的实验
        if exp_id in completed:
            print(f"\n[{exp_id}] 已完成，跳过")
            continue
        
        print(f"\n{'='*50}")
        print(f"[{exp_id}] 测试配置:")
        print(f"  Dropout: {dropout}")
        print(f"  LR_Classifier: {lr_clf}")
        print(f"  Weight_Decay: {wd}")
        print(f"{'='*50}")
        
        try:
            result = train_with_hyperparams(exp_id, dropout, lr_clf, wd)
            results.append(result)
            completed.add(exp_id)
            
            # 保存断点
            save_checkpoint(CHECKPOINT_FILE, {
                'completed': list(completed),
                'results': results
            })
            
            print(f"  ✓ Val Acc: {result['val_acc']:.4f}, Val F1: {result['val_f1']:.4f}")
            print(f"  ✓ 断点已保存")
            
        except Exception as e:
            print(f"  ❌ 实验失败: {e}")
            # 保存当前进度
            save_checkpoint(CHECKPOINT_FILE, {
                'completed': list(completed),
                'results': results
            })
            raise
    
    # 打印结果
    print("\n" + "="*70)
    print("超参数搜索结果")
    print("="*70)
    print(f"{'ID':<8} {'Dropout':<10} {'LR_Clf':<12} {'WD':<10} {'Val Acc':<10} {'Val F1':<10}")
    print("-"*70)
    
    for r in sorted(results, key=lambda x: -x['val_acc']):
        print(f"{r['exp_id']:<8} {r['dropout']:<10} {r['lr_classifier']:<12} {r['weight_decay']:<10} {r['val_acc']:.4f}     {r['val_f1']:.4f}")
    
    if results:
        best = max(results, key=lambda x: x['val_acc'])
        print(f"\n🏆 最佳配置:")
        print(f"   Dropout: {best['dropout']}")
        print(f"   LR_Classifier: {best['lr_classifier']}")
        print(f"   Weight_Decay: {best['weight_decay']}")
        print(f"   Val Acc: {best['val_acc']:.4f}")
    
    return results


def train_ensemble_models():
    """训练集成模型（多个种子，支持断点续传）"""
    
    print("\n" + "="*70)
    print("训练集成模型 (支持断点续传)")
    print("="*70)
    
    # 加载断点
    checkpoint = load_checkpoint(ENSEMBLE_CHECKPOINT_FILE)
    completed_seeds = set(checkpoint.get('completed_seeds', []))
    models = checkpoint.get('models', [])
    
    if completed_seeds:
        print(f"✓ 从断点恢复，已完成 {len(completed_seeds)} 个模型")
    
    seeds = [42, 123, 456]  # 3个不同种子
    
    for i, seed in enumerate(seeds):
        # 跳过已完成的
        if seed in completed_seeds:
            print(f"\n[{i+1}/{len(seeds)}] Seed={seed} 已完成，跳过")
            continue
        
        print(f"\n[{i+1}/{len(seeds)}] 训练模型 (seed={seed})")
        
        config = OPTIMIZED_CONFIG.copy()
        config['seed'] = seed
        set_seed(seed)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        train_loader, val_loader, _ = get_data_loaders(
            data_dir=config['data_dir'],
            train_label_file=config['train_label'],
            batch_size=config['batch_size'],
            val_ratio=config['val_ratio'],
            num_workers=0,
            seed=seed,
            force_resplit=True
        )
        
        model = OptimizedMultimodalClassifier(
            num_classes=3,
            feature_dim=config['feature_dim'],
            fusion_type='cross_attention',
            dropout=config['dropout'],
            unfreeze_text_layers=config['unfreeze_text_layers'],
            unfreeze_image_layers=config['unfreeze_image_layers']
        ).to(device)
        
        # 训练（简化版）
        class_weights = torch.FloatTensor([1.0, 1.5, 3.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        param_groups = model.get_param_groups(
            lr_pretrained=config['lr_pretrained'],
            lr_fusion=config['lr_fusion'],
            lr_classifier=config['lr_classifier']
        )
        optimizer = AdamW(param_groups, weight_decay=config['weight_decay'])
        
        total_steps = len(train_loader) * 20 // config['accumulation_steps']
        warmup_steps = int(total_steps * config['warmup_ratio'])
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        
        early_stopping = EarlyStopping(patience=7, mode='max')
        
        best_val_acc = 0
        
        for epoch in range(20):
            # 训练
            model.train()
            for step, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)):
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
            print(f"  Epoch {epoch+1}: Val Acc={val_metrics['accuracy']:.4f}")
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(model.state_dict(), f'experiments/checkpoints/ENSEMBLE_{i+1}_seed{seed}_best.pth')
                print(f"    ✓ 新最佳! Val Acc={best_val_acc:.4f}")
            
            if early_stopping(val_metrics['accuracy'], epoch):
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        models.append({
            'seed': seed,
            'val_acc': best_val_acc,
            'model_path': f'experiments/checkpoints/ENSEMBLE_{i+1}_seed{seed}_best.pth'
        })
        completed_seeds.add(seed)
        
        # 保存断点
        save_checkpoint(ENSEMBLE_CHECKPOINT_FILE, {
            'completed_seeds': list(completed_seeds),
            'models': models
        })
        print(f"  ✓ 断点已保存")
    
    print("\n" + "="*70)
    print("集成模型训练完成")
    print("="*70)
    for m in models:
        print(f"  Seed {m['seed']}: Val Acc={m['val_acc']:.4f}")
    
    if models:
        avg_acc = np.mean([m['val_acc'] for m in models])
        print(f"\n平均准确率: {avg_acc:.4f}")
    
    return models


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='务实优化方案')
    parser.add_argument('--hyperparam', action='store_true', help='超参数搜索')
    parser.add_argument('--ensemble', action='store_true', help='训练集成模型')
    
    args = parser.parse_args()
    
    if args.hyperparam:
        hyperparam_search()
    elif args.ensemble:
        train_ensemble_models()
    else:
        print("使用方法:")
        print("  py -3.11 run_practical_optimization.py --hyperparam  # 超参数搜索")
        print("  py -3.11 run_practical_optimization.py --ensemble    # 训练集成模型")


if __name__ == '__main__':
    main()
