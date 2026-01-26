"""
优化版实验脚本
核心优化：解冻编码器最后几层 + 分层学习率

关键改进：
1. 解冻DistilBERT最后2层 - 让文本特征适应任务
2. 解冻ResNet最后1个block - 让图像特征适应任务  
3. 分层学习率 - 预训练层用小学习率，新层用大学习率
4. 更强的类别平衡
"""
import os
import sys
import time
import json
import csv
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from data.data_loader import get_data_loaders
from utils.train_utils import set_seed, compute_metrics, EarlyStopping


# ========== 优化后的配置 ==========
OPTIMIZED_CONFIG = {
    'seed': 42,
    'val_ratio': 0.2,
    'batch_size': 8,
    'accumulation_steps': 4,
    'num_epochs': 30,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'early_stopping_patience': 7,  # 增加耐心
    'feature_dim': 512,
    'dropout': 0.3,
    'data_dir': r'D:\当代人工智能\project5\data',
    'train_label': r'D:\当代人工智能\project5\train.txt',
    
    # 关键优化参数
    'unfreeze_text_layers': 2,    # 解冻DistilBERT最后2层
    'unfreeze_image_layers': 1,   # 解冻ResNet最后1个block
    'lr_pretrained': 1e-5,        # 预训练层：小学习率
    'lr_fusion': 5e-5,            # 融合层：中学习率
    'lr_classifier': 1e-4,        # 分类器：大学习率
}


def get_results_csv_path():
    return 'experiments/all_results.csv'


def init_results_csv():
    os.makedirs('experiments', exist_ok=True)
    csv_path = get_results_csv_path()
    
    if not os.path.exists(csv_path):
        headers = [
            'exp_id', 'exp_name', 'modality', 'fusion_type',
            'val_acc', 'val_f1', 'val_loss',
            'train_acc', 'train_f1', 
            'best_epoch', 'total_epochs',
            'trainable_params', 'total_params',
            'training_time_min',
            'timestamp'
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    return csv_path


def save_result_to_csv(result: dict):
    csv_path = init_results_csv()
    
    row = [
        result.get('exp_id', ''),
        result.get('exp_name', ''),
        result.get('modality', ''),
        result.get('fusion_type', ''),
        f"{result.get('val_acc', 0):.4f}",
        f"{result.get('val_f1', 0):.4f}",
        f"{result.get('val_loss', 0):.4f}",
        f"{result.get('train_acc', 0):.4f}",
        f"{result.get('train_f1', 0):.4f}",
        result.get('best_epoch', 0),
        result.get('total_epochs', 0),
        result.get('trainable_params', 0),
        result.get('total_params', 0),
        f"{result.get('training_time_min', 0):.2f}",
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ]
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)
    
    print(f"✓ 结果已保存到: {csv_path}")


class OptimizedMultimodalClassifier(nn.Module):
    """
    优化版多模态分类器
    关键改进：解冻编码器最后几层
    """
    def __init__(
        self,
        num_classes=3,
        feature_dim=512,
        fusion_type='late',
        dropout=0.3,
        unfreeze_text_layers=2,
        unfreeze_image_layers=1
    ):
        super().__init__()
        
        from transformers import DistilBertModel
        import torchvision.models as models
        
        self.fusion_type = fusion_type
        
        # ========== 文本编码器 ==========
        print("加载 DistilBERT...")
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # 冻结大部分层，只解冻最后几层
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # 解冻最后 unfreeze_text_layers 层
        if unfreeze_text_layers > 0:
            for layer in self.text_encoder.transformer.layer[-unfreeze_text_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"  ✓ 解冻 DistilBERT 最后 {unfreeze_text_layers} 层")
        
        self.text_proj = nn.Sequential(
            nn.Linear(768, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ========== 图像编码器 ==========
        print("加载 ResNet50...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 冻结大部分层
        for param in resnet.parameters():
            param.requires_grad = False
        
        # 解冻最后 unfreeze_image_layers 个 block
        if unfreeze_image_layers >= 1:
            for param in resnet.layer4.parameters():
                param.requires_grad = True
            print(f"  ✓ 解冻 ResNet layer4")
        if unfreeze_image_layers >= 2:
            for param in resnet.layer3.parameters():
                param.requires_grad = True
            print(f"  ✓ 解冻 ResNet layer3")
        
        # 移除最后的FC层
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        self.image_proj = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ========== 融合层 ==========
        if fusion_type == 'late':
            fusion_dim = feature_dim * 2
            self.fusion = lambda t, i: torch.cat([t, i], dim=-1)
        elif fusion_type == 'cross_attention':
            self.cross_attn = nn.MultiheadAttention(feature_dim, num_heads=8, dropout=dropout, batch_first=True)
            self.cross_norm = nn.LayerNorm(feature_dim)
            fusion_dim = feature_dim * 2
        else:
            fusion_dim = feature_dim * 2
            self.fusion = lambda t, i: torch.cat([t, i], dim=-1)
        
        # ========== 分类器 ==========
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        print(f"  ✓ 融合类型: {fusion_type}, 融合维度: {fusion_dim}")
    
    def forward(self, input_ids, attention_mask, images):
        # 文本特征
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_output.last_hidden_state[:, 0, :]  # [CLS] token
        text_feat = self.text_proj(text_feat)
        
        # 图像特征
        image_feat = self.image_encoder(images)
        image_feat = image_feat.view(image_feat.size(0), -1)
        image_feat = self.image_proj(image_feat)
        
        # 融合
        if self.fusion_type == 'cross_attention':
            # 跨模态注意力
            text_seq = text_feat.unsqueeze(1)
            image_seq = image_feat.unsqueeze(1)
            attended, _ = self.cross_attn(text_seq, image_seq, image_seq)
            text_enhanced = self.cross_norm(text_feat + attended.squeeze(1))
            fused = torch.cat([text_enhanced, image_feat], dim=-1)
        else:
            fused = self.fusion(text_feat, image_feat)
        
        # 分类
        logits = self.classifier(fused)
        return logits
    
    def get_param_groups(self, lr_pretrained, lr_fusion, lr_classifier):
        """
        分层学习率：不同层使用不同学习率
        """
        pretrained_params = []
        fusion_params = []
        classifier_params = []
        
        # 文本编码器（预训练）
        for name, param in self.text_encoder.named_parameters():
            if param.requires_grad:
                pretrained_params.append(param)
        
        # 图像编码器（预训练）
        for name, param in self.image_encoder.named_parameters():
            if param.requires_grad:
                pretrained_params.append(param)
        
        # 投影层和融合层
        fusion_params.extend(self.text_proj.parameters())
        fusion_params.extend(self.image_proj.parameters())
        if hasattr(self, 'cross_attn'):
            fusion_params.extend(self.cross_attn.parameters())
            fusion_params.extend(self.cross_norm.parameters())
        
        # 分类器
        classifier_params.extend(self.classifier.parameters())
        
        param_groups = [
            {'params': pretrained_params, 'lr': lr_pretrained, 'name': 'pretrained'},
            {'params': fusion_params, 'lr': lr_fusion, 'name': 'fusion'},
            {'params': classifier_params, 'lr': lr_classifier, 'name': 'classifier'},
        ]
        
        return param_groups


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, accumulation_steps, tokenizer):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for step, batch in enumerate(tqdm(loader, desc='Training', leave=False)):
        texts = batch['text']
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Tokenize
        encoded = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        logits = model(input_ids, attention_mask, images)
        
        loss = criterion(logits, labels) / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), OPTIMIZED_CONFIG['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(loader)
    return metrics


def evaluate(model, loader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            texts = batch['text']
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            encoded = tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask, images)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(loader)
    return metrics


def run_optimized_experiment(exp_id: str, exp_name: str, fusion_type: str = 'late'):
    """运行优化版实验"""
    from transformers import DistilBertTokenizer
    
    print("\n" + "="*70)
    print(f"优化版实验 {exp_id}: {exp_name}")
    print(f"融合类型: {fusion_type}")
    print("关键优化: 解冻编码器 + 分层学习率")
    print("="*70)
    
    config = OPTIMIZED_CONFIG
    set_seed(config['seed'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, _ = get_data_loaders(
        data_dir=config['data_dir'],
        train_label_file=config['train_label'],
        batch_size=config['batch_size'],
        val_ratio=config['val_ratio'],
        num_workers=0,
        seed=config['seed'],
        force_resplit=True
    )
    print(f"训练集: {len(train_loader.dataset)}, 验证集: {len(val_loader.dataset)}")
    
    # 加载tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 创建模型
    print("\n创建优化版模型...")
    model = OptimizedMultimodalClassifier(
        num_classes=3,
        feature_dim=config['feature_dim'],
        fusion_type=fusion_type,
        dropout=config['dropout'],
        unfreeze_text_layers=config['unfreeze_text_layers'],
        unfreeze_image_layers=config['unfreeze_image_layers']
    ).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}, 可训练: {trainable_params:,}")
    
    # 类别权重（加强neutral）
    train_labels = np.concatenate([b['label'].numpy() for b in train_loader])
    class_counts = np.bincount(train_labels)
    # 更强的权重给少数类
    class_weights = torch.FloatTensor([1.0, 1.5, 3.0]).to(device)  # positive, negative, neutral
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 分层学习率优化器
    param_groups = model.get_param_groups(
        lr_pretrained=config['lr_pretrained'],
        lr_fusion=config['lr_fusion'],
        lr_classifier=config['lr_classifier']
    )
    optimizer = AdamW(param_groups, weight_decay=config['weight_decay'])
    
    print(f"\n分层学习率:")
    print(f"  预训练层: {config['lr_pretrained']}")
    print(f"  融合层: {config['lr_fusion']}")
    print(f"  分类器: {config['lr_classifier']}")
    
    total_steps = len(train_loader) * config['num_epochs'] // config['accumulation_steps']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'], mode='max')
    
    # 训练
    best_val_acc = 0
    best_val_f1 = 0
    best_val_loss = float('inf')
    best_epoch = 0
    best_train_acc = 0
    best_train_f1 = 0
    start_time = time.time()
    
    print("\n开始训练...")
    for epoch in range(config['num_epochs']):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, config['accumulation_steps'], tokenizer
        )
        val_metrics = evaluate(model, val_loader, criterion, device, tokenizer)
        
        print(f"Epoch {epoch+1:2d}: "
              f"Train Acc={train_metrics['accuracy']:.4f}, "
              f"Val Acc={val_metrics['accuracy']:.4f}, "
              f"Val F1={val_metrics['f1']:.4f}")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_val_f1 = val_metrics['f1']
            best_val_loss = val_metrics['loss']
            best_epoch = epoch + 1
            best_train_acc = train_metrics['accuracy']
            best_train_f1 = train_metrics['f1']
            
            os.makedirs('experiments/checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'experiments/checkpoints/{exp_id}_{exp_name}_best.pth')
            print(f"  ✓ 新最佳! 已保存模型")
        
        if early_stopping(val_metrics['accuracy'], epoch):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = (time.time() - start_time) / 60
    
    result = {
        'exp_id': exp_id,
        'exp_name': exp_name,
        'modality': 'multimodal_optimized',
        'fusion_type': fusion_type,
        'val_acc': best_val_acc,
        'val_f1': best_val_f1,
        'val_loss': best_val_loss,
        'train_acc': best_train_acc,
        'train_f1': best_train_f1,
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'training_time_min': training_time
    }
    
    save_result_to_csv(result)
    
    print(f"\n实验 {exp_id} 完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"训练时间: {training_time:.1f} 分钟")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='优化版多模态实验')
    parser.add_argument('--fusion', type=str, default='late', choices=['late', 'cross_attention'],
                       help='融合类型')
    parser.add_argument('--all', action='store_true', help='运行所有优化实验')
    
    args = parser.parse_args()
    
    if args.all:
        print("\n" + "="*70)
        print("运行所有优化版实验")
        print("="*70)
        
        results = []
        for fusion in ['late', 'cross_attention']:
            result = run_optimized_experiment(f'OPT_{fusion}', f'optimized_{fusion}', fusion)
            results.append(result)
        
        print("\n" + "="*70)
        print("优化版实验结果对比")
        print("="*70)
        print(f"{'融合方法':<20} {'Val Acc':<10} {'Val F1':<10} {'可训练参数':<15}")
        print("-"*55)
        for r in results:
            print(f"{r['fusion_type']:<20} {r['val_acc']:.4f}     {r['val_f1']:.4f}     {r['trainable_params']:,}")
    else:
        run_optimized_experiment(f'OPT_{args.fusion}', f'optimized_{args.fusion}', args.fusion)


if __name__ == '__main__':
    main()
