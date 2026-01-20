"""
Training logger for visualization and analysis
记录训练过程中的所有关键指标，用于后续可视化和分析
"""

import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Any


class TrainingLogger:
    """记录训练过程中的所有关键数据"""
    
    def __init__(self, log_dir='logs', experiment_name=None):
        """
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称（如果None则自动生成）
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 实验名称
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f'exp_{timestamp}'
        self.experiment_name = experiment_name
        
        # 创建实验目录
        self.exp_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 历史记录
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_precision': [],
            'train_recall': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # 每个step的详细记录（可选，用于细粒度分析）
        self.step_logs = []
        
        # 最佳指标
        self.best_metrics = {
            'epoch': 0,
            'val_acc': 0.0,
            'val_f1': 0.0
        }
        
        # 配置信息
        self.config = {}
        
        # 错误样本记录
        self.error_samples = []
        
        print(f"✓ Logger initialized: {self.exp_dir}")
    
    def log_config(self, config_dict: Dict):
        """记录实验配置"""
        self.config = config_dict
        config_file = os.path.join(self.exp_dir, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"✓ Config saved: {config_file}")
    
    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict, 
                  lr: float, epoch_time: float):
        """
        记录每个epoch的指标
        
        Args:
            epoch: 当前epoch
            train_metrics: 训练集指标 {'loss', 'accuracy', 'precision', 'recall', 'f1'}
            val_metrics: 验证集指标
            lr: 当前学习率
            epoch_time: epoch耗时（秒）
        """
        # 添加到历史记录
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_acc'].append(train_metrics['accuracy'])
        self.history['train_precision'].append(train_metrics['precision'])
        self.history['train_recall'].append(train_metrics['recall'])
        self.history['train_f1'].append(train_metrics['f1'])
        
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_acc'].append(val_metrics['accuracy'])
        self.history['val_precision'].append(val_metrics['precision'])
        self.history['val_recall'].append(val_metrics['recall'])
        self.history['val_f1'].append(val_metrics['f1'])
        
        self.history['learning_rate'].append(lr)
        self.history['epoch_time'].append(epoch_time)
        
        # 更新最佳指标
        if val_metrics['accuracy'] > self.best_metrics['val_acc']:
            self.best_metrics = {
                'epoch': epoch,
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1']
            }
        
        # 保存历史记录
        self._save_history()
    
    def log_step(self, epoch: int, step: int, loss: float, lr: float):
        """
        记录每个训练step的详细信息（可选，用于细粒度分析）
        
        Args:
            epoch: 当前epoch
            step: 当前step
            loss: 当前loss
            lr: 当前学习率
        """
        self.step_logs.append({
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'lr': lr
        })
        
        # 每100步保存一次
        if len(self.step_logs) % 100 == 0:
            self._save_step_logs()
    
    def log_error_samples(self, guids: List[str], true_labels: List[int], 
                         pred_labels: List[int], epoch: int = None):
        """
        记录预测错误的样本
        
        Args:
            guids: 样本ID列表
            true_labels: 真实标签
            pred_labels: 预测标签
            epoch: 所属epoch（可选）
        """
        for guid, true_label, pred_label in zip(guids, true_labels, pred_labels):
            if true_label != pred_label:
                self.error_samples.append({
                    'guid': guid,
                    'true_label': int(true_label),
                    'pred_label': int(pred_label),
                    'epoch': epoch
                })
        
        # 保存错误样本
        self._save_error_samples()
    
    def _save_history(self):
        """保存训练历史到JSON和CSV"""
        # JSON格式（完整数据）
        history_file = os.path.join(self.exp_dir, 'history.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump({
                'history': self.history,
                'best_metrics': self.best_metrics
            }, f, indent=2, ensure_ascii=False)
        
        # CSV格式（方便Excel打开）
        csv_file = os.path.join(self.exp_dir, 'history.csv')
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 表头
            writer.writerow([
                'epoch', 'train_loss', 'train_acc', 'train_f1',
                'val_loss', 'val_acc', 'val_f1', 'learning_rate', 'epoch_time'
            ])
            # 数据
            num_epochs = len(self.history['train_loss'])
            for i in range(num_epochs):
                writer.writerow([
                    i + 1,
                    self.history['train_loss'][i],
                    self.history['train_acc'][i],
                    self.history['train_f1'][i],
                    self.history['val_loss'][i],
                    self.history['val_acc'][i],
                    self.history['val_f1'][i],
                    self.history['learning_rate'][i],
                    self.history['epoch_time'][i]
                ])
    
    def _save_step_logs(self):
        """保存step级别的日志"""
        step_file = os.path.join(self.exp_dir, 'step_logs.csv')
        with open(step_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'loss', 'learning_rate'])
            for log in self.step_logs:
                writer.writerow([
                    log['epoch'], log['step'], log['loss'], log['lr']
                ])
    
    def _save_error_samples(self):
        """保存错误样本记录"""
        error_file = os.path.join(self.exp_dir, 'error_samples.csv')
        with open(error_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['guid', 'true_label', 'pred_label', 'epoch'])
            for sample in self.error_samples:
                writer.writerow([
                    sample['guid'],
                    sample['true_label'],
                    sample['pred_label'],
                    sample.get('epoch', '')
                ])
    
    def save_final_summary(self):
        """保存最终实验摘要"""
        summary = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'best_metrics': self.best_metrics,
            'total_epochs': len(self.history['train_loss']),
            'final_train_acc': self.history['train_acc'][-1] if self.history['train_acc'] else 0,
            'final_val_acc': self.history['val_acc'][-1] if self.history['val_acc'] else 0,
            'total_training_time': sum(self.history['epoch_time']),
            'num_error_samples': len(self.error_samples)
        }
        
        summary_file = os.path.join(self.exp_dir, 'summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Best Epoch: {summary['best_metrics']['epoch']}")
        print(f"Best Val Acc: {summary['best_metrics']['val_acc']:.4f}")
        print(f"Best Val F1: {summary['best_metrics']['val_f1']:.4f}")
        print(f"Total Training Time: {summary['total_training_time']:.1f}s")
        print(f"Results saved to: {self.exp_dir}")
        print(f"{'='*60}\n")
    
    def get_history(self):
        """获取训练历史"""
        return self.history
    
    def get_best_metrics(self):
        """获取最佳指标"""
        return self.best_metrics


class ExperimentTracker:
    """
    跟踪多个实验的对比
    用于消融实验、融合策略对比等
    """
    
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.experiments = {}
        self.comparison_file = os.path.join(log_dir, 'experiments_comparison.csv')
    
    def add_experiment(self, name: str, config: Dict, best_metrics: Dict):
        """
        添加一个实验结果
        
        Args:
            name: 实验名称（如 'baseline_late_fusion', 'text_only', 'early_fusion'）
            config: 实验配置
            best_metrics: 最佳指标
        """
        self.experiments[name] = {
            'config': config,
            'best_metrics': best_metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self._save_comparison()
    
    def _save_comparison(self):
        """保存实验对比表"""
        with open(self.comparison_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 表头
            writer.writerow([
                'experiment_name', 'fusion_type', 'feature_dim',
                'batch_size', 'learning_rate', 'best_epoch',
                'best_val_acc', 'best_val_f1', 'timestamp'
            ])
            # 数据
            for name, exp in self.experiments.items():
                writer.writerow([
                    name,
                    exp['config'].get('fusion_type', 'N/A'),
                    exp['config'].get('feature_dim', 'N/A'),
                    exp['config'].get('batch_size', 'N/A'),
                    exp['config'].get('learning_rate', 'N/A'),
                    exp['best_metrics'].get('epoch', 'N/A'),
                    exp['best_metrics'].get('val_acc', 0),
                    exp['best_metrics'].get('val_f1', 0),
                    exp['timestamp']
                ])
        
        print(f"✓ Comparison saved: {self.comparison_file}")
