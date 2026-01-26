"""
简化版实验运行脚本
专注于核心实验，避免过度复杂化
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
from models.multimodal_model import MultimodalClassifier, TextOnlyClassifier, ImageOnlyClassifier
from utils.train_utils import set_seed, count_parameters, compute_metrics, EarlyStopping


# ========== 固定实验配置（确保公平对比） ==========
FIXED_CONFIG = {
    'seed': 42,
    'val_ratio': 0.2,           # 80/20划分
    'batch_size': 8,
    'accumulation_steps': 4,    # effective batch = 32
    'num_epochs': 30,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'early_stopping_patience': 5,
    'feature_dim': 512,
    'dropout': 0.3,
    'freeze_encoders': True,
    'data_dir': r'D:\当代人工智能\project5\data',
    'train_label': r'D:\当代人工智能\project5\train.txt',
}


def get_results_csv_path():
    """获取结果CSV路径"""
    return 'experiments/all_results.csv'


def init_results_csv():
    """初始化结果CSV文件"""
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
    """保存单次实验结果到CSV"""
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


def create_model(modality: str, fusion_type: str, device: str):
    """创建模型"""
    config = FIXED_CONFIG
    
    if modality == 'text':
        model = TextOnlyClassifier(
            num_classes=3,
            text_model='distilbert-base-uncased',
            feature_dim=config['feature_dim'],
            dropout=config['dropout']
        )
    elif modality == 'image':
        model = ImageOnlyClassifier(
            num_classes=3,
            image_model='resnet50',
            feature_dim=config['feature_dim'],
            dropout=config['dropout']
        )
    else:  # multimodal
        model = MultimodalClassifier(
            num_classes=3,
            text_model='distilbert-base-uncased',
            image_model='resnet50',
            fusion_type=fusion_type,
            feature_dim=config['feature_dim'],
            freeze_encoders=config['freeze_encoders'],
            dropout=config['dropout']
        )
    
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, modality, accumulation_steps):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for step, batch in enumerate(tqdm(loader, desc='Training', leave=False)):
        texts = batch['text']
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        if modality == 'text':
            logits = model(texts)
        elif modality == 'image':
            logits = model(images)
        else:
            logits = model(texts, images)
        
        loss = criterion(logits, labels) / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), FIXED_CONFIG['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(loader)
    return metrics


def evaluate(model, loader, criterion, device, modality):
    """验证"""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            texts = batch['text']
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            if modality == 'text':
                logits = model(texts)
            elif modality == 'image':
                logits = model(images)
            else:
                logits = model(texts, images)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(loader)
    return metrics


def run_single_experiment(exp_id: str, exp_name: str, modality: str, fusion_type: str = 'late'):
    """
    运行单个实验
    """
    print("\n" + "="*70)
    print(f"实验 {exp_id}: {exp_name}")
    print(f"模态: {modality}, 融合: {fusion_type}")
    print("="*70)
    
    config = FIXED_CONFIG
    set_seed(config['seed'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 加载数据（强制重新划分以使用新的val_ratio）
    print("\n加载数据...")
    train_loader, val_loader, _ = get_data_loaders(
        data_dir=config['data_dir'],
        train_label_file=config['train_label'],
        batch_size=config['batch_size'],
        val_ratio=config['val_ratio'],
        num_workers=0,
        seed=config['seed'],
        force_resplit=True  # 强制使用新划分
    )
    print(f"训练集: {len(train_loader.dataset)}, 验证集: {len(val_loader.dataset)}")
    
    # 创建模型
    print("\n创建模型...")
    model = create_model(modality, fusion_type, device)
    total_params, trainable_params = count_parameters(model)
    print(f"总参数: {total_params:,}, 可训练: {trainable_params:,}")
    
    # 类别权重
    train_labels = np.concatenate([b['label'].numpy() for b in train_loader])
    class_counts = np.bincount(train_labels)
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    class_weights = class_weights / class_weights.sum() * 3
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    total_steps = len(train_loader) * config['num_epochs'] // config['accumulation_steps']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'], mode='max')
    
    # 训练
    best_val_acc = 0
    best_val_f1 = 0
    best_val_loss = float('inf')
    best_epoch = 0
    start_time = time.time()
    
    print("\n开始训练...")
    for epoch in range(config['num_epochs']):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, 
            device, modality, config['accumulation_steps']
        )
        val_metrics = evaluate(model, val_loader, criterion, device, modality)
        
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
            
            # 保存最佳模型
            os.makedirs('experiments/checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'experiments/checkpoints/{exp_id}_{exp_name}_best.pth')
            print(f"  ✓ 新最佳! 已保存模型")
        
        if early_stopping(val_metrics['accuracy'], epoch):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = (time.time() - start_time) / 60
    
    # 保存结果
    result = {
        'exp_id': exp_id,
        'exp_name': exp_name,
        'modality': modality,
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


def run_phase1_ablation():
    """阶段1: 消融实验"""
    print("\n" + "="*70)
    print("阶段1: 消融实验 (Ablation Study)")
    print("="*70)
    
    experiments = [
        ('A1', 'multimodal_late', 'multimodal', 'late'),
        ('A2', 'text_only', 'text', 'none'),
        ('A3', 'image_only', 'image', 'none'),
    ]
    
    results = []
    for exp_id, exp_name, modality, fusion in experiments:
        result = run_single_experiment(exp_id, exp_name, modality, fusion)
        results.append(result)
    
    # 打印消融实验对比
    print("\n" + "="*70)
    print("消融实验结果对比")
    print("="*70)
    print(f"{'实验':<20} {'模态':<12} {'Val Acc':<10} {'Val F1':<10}")
    print("-"*52)
    for r in results:
        print(f"{r['exp_name']:<20} {r['modality']:<12} {r['val_acc']:.4f}     {r['val_f1']:.4f}")
    
    return results


def run_phase2_fusion():
    """阶段2: 融合策略对比"""
    print("\n" + "="*70)
    print("阶段2: 融合策略对比 (Fusion Comparison)")
    print("="*70)
    
    experiments = [
        ('F1', 'fusion_late', 'multimodal', 'late'),
        ('F2', 'fusion_early', 'multimodal', 'early'),
        ('F3', 'fusion_cross_attn', 'multimodal', 'cross_attention'),
        ('F4', 'fusion_gated', 'multimodal', 'gated'),
    ]
    
    results = []
    for exp_id, exp_name, modality, fusion in experiments:
        result = run_single_experiment(exp_id, exp_name, modality, fusion)
        results.append(result)
    
    print("\n" + "="*70)
    print("融合策略对比结果")
    print("="*70)
    print(f"{'融合方法':<20} {'Val Acc':<10} {'Val F1':<10} {'Best Epoch':<10}")
    print("-"*50)
    for r in results:
        print(f"{r['fusion_type']:<20} {r['val_acc']:.4f}     {r['val_f1']:.4f}     {r['best_epoch']}")
    
    return results


def run_phase3_advanced():
    """阶段3: 高级融合方法（解决空间不对齐问题）"""
    print("\n" + "="*70)
    print("阶段3: 高级融合方法 (Advanced Fusion - 解决模态空间不对齐)")
    print("="*70)
    
    experiments = [
        ('AF1', 'aligned_fusion', 'multimodal', 'aligned'),
        ('AF2', 'hierarchical_fusion', 'multimodal', 'hierarchical'),
    ]
    
    results = []
    for exp_id, exp_name, modality, fusion in experiments:
        result = run_single_experiment(exp_id, exp_name, modality, fusion)
        results.append(result)
    
    print("\n" + "="*70)
    print("高级融合方法结果")
    print("="*70)
    print(f"{'融合方法':<25} {'Val Acc':<10} {'Val F1':<10} {'Best Epoch':<10}")
    print("-"*55)
    for r in results:
        print(f"{r['fusion_type']:<25} {r['val_acc']:.4f}     {r['val_f1']:.4f}     {r['best_epoch']}")
    
    return results


def print_all_results():
    """打印所有实验结果"""
    csv_path = get_results_csv_path()
    if not os.path.exists(csv_path):
        print("暂无实验结果")
        return
    
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*70)
    print("所有实验结果汇总")
    print("="*70)
    print(df[['exp_id', 'exp_name', 'modality', 'fusion_type', 'val_acc', 'val_f1']].to_string(index=False))
    
    if len(df) > 0:
        best = df.loc[df['val_acc'].astype(float).idxmax()]
        print(f"\n🏆 最佳实验: {best['exp_id']} - {best['exp_name']}")
        print(f"   Val Acc: {best['val_acc']}, Val F1: {best['val_f1']}")


def main():
    parser = argparse.ArgumentParser(description='多模态情感分类实验')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], help='运行实验阶段 (1=消融, 2=基础融合对比, 3=高级融合)')
    parser.add_argument('--exp', type=str, help='运行单个实验，格式: modality,fusion (如: multimodal,late)')
    parser.add_argument('--results', action='store_true', help='显示所有结果')
    
    args = parser.parse_args()
    
    if args.results:
        print_all_results()
        return
    
    if args.phase == 1:
        run_phase1_ablation()
    elif args.phase == 2:
        run_phase2_fusion()
    elif args.phase == 3:
        run_phase3_advanced()
    elif args.exp:
        parts = args.exp.split(',')
        modality = parts[0]
        fusion = parts[1] if len(parts) > 1 else 'late'
        run_single_experiment('custom', f'{modality}_{fusion}', modality, fusion)
    else:
        print("使用方法:")
        print("  python run_experiment_simple.py --phase 1    # 运行消融实验 (text/image/multimodal)")
        print("  python run_experiment_simple.py --phase 2    # 运行基础融合对比 (late/early/cross_attn/gated)")
        print("  python run_experiment_simple.py --phase 3    # 运行高级融合 (aligned/hierarchical)")
        print("  python run_experiment_simple.py --results    # 查看结果")
        print("  python run_experiment_simple.py --exp multimodal,aligned  # 单个实验")


if __name__ == '__main__':
    main()
