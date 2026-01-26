"""
统一实验运行脚本
支持运行预定义实验或自定义实验
确保所有实验使用相同的配置框架，结果可对比
"""
import os
import sys
import time
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from configs.experiment_config import ExperimentConfig, get_experiment, list_experiments, EXPERIMENTS
from data.data_loader import get_data_loaders
from models.multimodal_model import MultimodalClassifier, TextOnlyClassifier, ImageOnlyClassifier
from utils.train_utils import set_seed, count_parameters, save_checkpoint, compute_metrics, EarlyStopping
from utils.experiment_logger import ExperimentLogger, TrainingHistoryRecorder


def create_model(config: dict, device: str):
    """根据配置创建模型"""
    modality = config['modality']
    fusion_type = config['fusion_type']
    
    if modality == 'text':
        model = TextOnlyClassifier(
            num_classes=config['num_classes'],
            text_model=config['text_model'],
            feature_dim=config['feature_dim'],
            dropout=config['dropout']
        )
        model_type = 'text_only'
    elif modality == 'image':
        model = ImageOnlyClassifier(
            num_classes=config['num_classes'],
            image_model=config['image_model'],
            feature_dim=config['feature_dim'],
            dropout=config['dropout']
        )
        model_type = 'image_only'
    else:  # multimodal
        model = MultimodalClassifier(
            num_classes=config['num_classes'],
            text_model=config['text_model'],
            image_model=config['image_model'],
            fusion_type=fusion_type,
            feature_dim=config['feature_dim'],
            freeze_encoders=config['freeze_encoders'],
            dropout=config['dropout']
        )
        model_type = 'multimodal'
    
    return model.to(device), model_type


def train_epoch(model, train_loader, criterion, optimizer, scheduler, 
                config, epoch, model_type='multimodal'):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    device = config.get('device', 'cpu')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for step, batch in enumerate(pbar):
        texts = batch['text']
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward
        if model_type == 'text_only':
            logits = model(texts)
        elif model_type == 'image_only':
            logits = model(images)
        else:
            logits = model(texts, images)
        
        loss = criterion(logits, labels)
        loss = loss / config['accumulation_steps']
        loss.backward()
        
        if (step + 1) % config['accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config['accumulation_steps']
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item() * config["accumulation_steps"]:.4f}'})
    
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(train_loader)
    return metrics


def evaluate(model, val_loader, criterion, config, model_type='multimodal'):
    """验证"""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    device = config.get('device', 'cpu')
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            texts = batch['text']
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            if model_type == 'text_only':
                logits = model(texts)
            elif model_type == 'image_only':
                logits = model(images)
            else:
                logits = model(texts, images)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(val_loader)
    return metrics


def run_experiment(config: dict, verbose: bool = True):
    """
    运行单个实验
    
    Args:
        config: 实验配置字典
        verbose: 是否打印详细信息
    
    Returns:
        results: 实验结果字典
    """
    exp_id = config['exp_id']
    exp_name = config['exp_name']
    
    print("\n" + "="*70)
    print(f"实验 {exp_id}: {exp_name}")
    print("="*70)
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['device'] = device
    print(f"使用设备: {device}")
    
    # 初始化日志器
    exp_logger = ExperimentLogger(config['experiment_dir'])
    exp_folder = exp_logger.create_experiment_folder(exp_id, exp_name)
    exp_logger.save_config(exp_folder, config)
    history_recorder = TrainingHistoryRecorder(exp_folder)
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, _ = get_data_loaders(
        data_dir=config['data_dir'],
        train_label_file=config['train_label'],
        batch_size=config['batch_size'],
        val_ratio=config['val_ratio'],
        num_workers=0,
        seed=config['seed'],
        enhanced_augmentation=config.get('use_augmentation', False)
    )
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    
    # 创建模型
    print("\n创建模型...")
    model, model_type = create_model(config, device)
    total_params, trainable_params = count_parameters(model)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # 类别权重
    train_labels = [batch['label'].numpy() for batch in train_loader]
    train_labels = np.concatenate(train_labels)
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度
    total_steps = len(train_loader) * config['num_epochs'] // config['accumulation_steps']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'], mode='max')
    
    # 训练
    best_val_acc = 0
    best_epoch = 0
    train_start = time.time()
    
    print("\n开始训练...")
    for epoch in range(config['num_epochs']):
        epoch_start = time.time()
        history_recorder.start_epoch(epoch + 1)
        
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, 
                                    scheduler, config, epoch, model_type)
        history_recorder.log_train_metrics(
            train_metrics['loss'], train_metrics['accuracy'], train_metrics['f1']
        )
        
        # 验证
        val_metrics = evaluate(model, val_loader, criterion, config, model_type)
        history_recorder.log_val_metrics(
            val_metrics['loss'], val_metrics['accuracy'], val_metrics['f1'],
            val_metrics['precision'], val_metrics['recall']
        )
        
        history_recorder.log_lr(optimizer.param_groups[0]['lr'])
        history_recorder.end_epoch(time.time() - epoch_start)
        
        if verbose:
            print(f"\nEpoch {epoch+1}/{config['num_epochs']}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            best_val_f1 = val_metrics['f1']
            best_val_precision = val_metrics['precision']
            best_val_recall = val_metrics['recall']
            
            # 保存模型
            model_path = os.path.join(exp_folder, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': config
            }, model_path)
            print(f"  ✓ 新最佳模型! Acc: {best_val_acc:.4f}")
        
        # Early stopping
        if early_stopping(val_metrics['accuracy'], epoch):
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - train_start
    
    # 保存训练历史
    history_recorder.save()
    
    # 汇总结果
    results = {
        'val_acc': best_val_acc,
        'val_f1': best_val_f1,
        'val_precision': best_val_precision,
        'val_recall': best_val_recall,
        'train_acc': train_metrics['accuracy'],
        'train_f1': train_metrics['f1'],
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'total_time_sec': total_time,
        'trainable_params': trainable_params,
        'total_params': total_params
    }
    
    # 保存评估结果
    exp_logger.save_evaluation_results(exp_folder, results)
    
    # 记录到汇总表
    exp_logger.log_experiment(config, results)
    
    print("\n" + "="*70)
    print(f"实验 {exp_id} 完成!")
    print(f"  最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"  最佳验证F1: {best_val_f1:.4f}")
    print(f"  总训练时间: {total_time/60:.2f} 分钟")
    print(f"  结果保存至: {exp_folder}")
    print("="*70)
    
    return results


def run_all_experiments(exp_ids: list = None):
    """
    运行多个实验
    
    Args:
        exp_ids: 要运行的实验ID列表，None则运行所有
    """
    if exp_ids is None:
        exp_ids = list(EXPERIMENTS.keys())
    
    print("\n" + "="*70)
    print(f"批量运行实验: {exp_ids}")
    print("="*70)
    
    all_results = {}
    
    for exp_id in exp_ids:
        try:
            config = get_experiment(exp_id)
            results = run_experiment(config)
            all_results[exp_id] = results
        except Exception as e:
            print(f"\n❌ 实验 {exp_id} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印汇总
    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)
    
    logger = ExperimentLogger()
    logger.print_summary()
    
    # 生成对比表格
    logger.generate_comparison_table('experiments/comparison.md')
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='运行多模态情感分类实验')
    parser.add_argument('--exp', type=str, default=None,
                        help='实验ID (如 E1.1)，不指定则列出所有实验')
    parser.add_argument('--list', action='store_true',
                        help='列出所有预定义实验')
    parser.add_argument('--run-all', action='store_true',
                        help='运行所有实验')
    parser.add_argument('--run-phase', type=int, choices=[1, 2, 3, 4],
                        help='运行指定阶段的实验 (1=消融, 2=融合对比, 3=增强对比, 4=改进)')
    parser.add_argument('--summary', action='store_true',
                        help='显示实验汇总')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    if args.summary:
        logger = ExperimentLogger()
        logger.print_summary()
        return
    
    if args.run_all:
        run_all_experiments()
        return
    
    if args.run_phase:
        phase_experiments = {
            1: ['E1.1', 'E1.2', 'E1.3'],
            2: ['E2.1', 'E2.2', 'E2.3', 'E2.4'],
            3: ['E3.1', 'E3.2', 'E3.3', 'E3.4'],
            4: ['E4.1', 'E4.2']
        }
        run_all_experiments(phase_experiments[args.run_phase])
        return
    
    if args.exp:
        config = get_experiment(args.exp)
        run_experiment(config)
        return
    
    # 默认列出实验
    list_experiments()
    print("使用示例:")
    print("  python run_experiments.py --exp E1.1      # 运行指定实验")
    print("  python run_experiments.py --run-phase 1  # 运行阶段1所有实验")
    print("  python run_experiments.py --run-all      # 运行所有实验")
    print("  python run_experiments.py --summary      # 查看实验汇总")


if __name__ == '__main__':
    main()
