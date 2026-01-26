"""
实验结果记录器
统一记录所有实验结果到CSV，确保可复现和对比
"""
import os
import json
import csv
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd


class ExperimentLogger:
    """
    实验日志记录器
    - 记录每个实验的配置和结果
    - 保存训练历史
    - 生成对比表格
    """
    
    def __init__(self, experiment_dir: str = 'experiments'):
        """
        Args:
            experiment_dir: 实验结果保存目录
        """
        self.experiment_dir = experiment_dir
        self.summary_file = os.path.join(experiment_dir, 'experiment_summary.csv')
        
        # 创建目录
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 初始化汇总CSV
        self._init_summary_csv()
    
    def _init_summary_csv(self):
        """初始化实验汇总CSV文件"""
        if not os.path.exists(self.summary_file):
            headers = [
                'exp_id', 'exp_name', 'fusion_type', 'modality',
                'text_model', 'image_model', 'use_augmentation',
                'val_acc', 'val_f1', 'val_precision', 'val_recall',
                'train_acc', 'train_f1',
                'best_epoch', 'total_epochs', 'total_time_min',
                'trainable_params', 'total_params',
                'seed', 'val_ratio', 'batch_size', 'learning_rate',
                'timestamp', 'notes'
            ]
            with open(self.summary_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            print(f"✓ 创建实验汇总文件: {self.summary_file}")
    
    def create_experiment_folder(self, exp_id: str, exp_name: str) -> str:
        """创建实验文件夹"""
        folder_name = f"{exp_id}_{exp_name}"
        exp_folder = os.path.join(self.experiment_dir, folder_name)
        os.makedirs(exp_folder, exist_ok=True)
        return exp_folder
    
    def save_config(self, exp_folder: str, config: Dict[str, Any]):
        """保存实验配置"""
        config_path = os.path.join(exp_folder, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
        print(f"✓ 配置已保存: {config_path}")
    
    def save_training_history(self, exp_folder: str, history: list):
        """
        保存训练历史到CSV
        
        Args:
            exp_folder: 实验文件夹
            history: 训练历史列表，每个元素是一个epoch的dict
        """
        history_path = os.path.join(exp_folder, 'training_history.csv')
        
        if not history:
            print("⚠️ 训练历史为空")
            return
        
        # 获取所有键作为列名
        fieldnames = list(history[0].keys())
        
        with open(history_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        
        print(f"✓ 训练历史已保存: {history_path}")
    
    def save_evaluation_results(self, exp_folder: str, results: Dict[str, Any]):
        """保存评估结果"""
        results_path = os.path.join(exp_folder, 'evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"✓ 评估结果已保存: {results_path}")
    
    def log_experiment(self, 
                       config: Dict[str, Any],
                       results: Dict[str, Any],
                       notes: str = ''):
        """
        记录完整实验到汇总CSV
        
        Args:
            config: 实验配置
            results: 实验结果
            notes: 备注
        """
        row = {
            'exp_id': config.get('exp_id', ''),
            'exp_name': config.get('exp_name', ''),
            'fusion_type': config.get('fusion_type', ''),
            'modality': config.get('modality', ''),
            'text_model': config.get('text_model', ''),
            'image_model': config.get('image_model', ''),
            'use_augmentation': config.get('use_augmentation', False),
            'val_acc': results.get('val_acc', 0),
            'val_f1': results.get('val_f1', 0),
            'val_precision': results.get('val_precision', 0),
            'val_recall': results.get('val_recall', 0),
            'train_acc': results.get('train_acc', 0),
            'train_f1': results.get('train_f1', 0),
            'best_epoch': results.get('best_epoch', 0),
            'total_epochs': results.get('total_epochs', 0),
            'total_time_min': results.get('total_time_sec', 0) / 60,
            'trainable_params': results.get('trainable_params', 0),
            'total_params': results.get('total_params', 0),
            'seed': config.get('seed', 42),
            'val_ratio': config.get('val_ratio', 0.2),
            'batch_size': config.get('batch_size', 8),
            'learning_rate': config.get('learning_rate', 2e-5),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'notes': notes
        }
        
        # 追加到CSV
        with open(self.summary_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
        
        print(f"✓ 实验 {config.get('exp_id')} 已记录到汇总表")
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """获取实验汇总DataFrame"""
        if os.path.exists(self.summary_file):
            return pd.read_csv(self.summary_file)
        return pd.DataFrame()
    
    def print_summary(self):
        """打印实验汇总表"""
        df = self.get_summary_dataframe()
        
        if df.empty:
            print("暂无实验记录")
            return
        
        print("\n" + "="*100)
        print("实验结果汇总")
        print("="*100)
        
        # 选择关键列显示
        display_cols = ['exp_id', 'exp_name', 'fusion_type', 'modality', 
                       'val_acc', 'val_f1', 'best_epoch']
        
        if all(col in df.columns for col in display_cols):
            display_df = df[display_cols].copy()
            display_df['val_acc'] = display_df['val_acc'].apply(lambda x: f"{x:.4f}")
            display_df['val_f1'] = display_df['val_f1'].apply(lambda x: f"{x:.4f}")
            print(display_df.to_string(index=False))
        else:
            print(df.to_string())
        
        print("="*100)
        
        # 找出最佳实验
        if 'val_acc' in df.columns and len(df) > 0:
            best_idx = df['val_acc'].idxmax()
            best_exp = df.loc[best_idx]
            print(f"\n🏆 最佳实验: {best_exp['exp_id']} - {best_exp['exp_name']}")
            print(f"   验证集准确率: {best_exp['val_acc']:.4f}")
            print(f"   验证集F1: {best_exp['val_f1']:.4f}")
    
    def generate_comparison_table(self, output_path: str = None) -> str:
        """
        生成Markdown格式的对比表格
        
        Args:
            output_path: 输出路径，None则返回字符串
        """
        df = self.get_summary_dataframe()
        
        if df.empty:
            return "暂无实验数据"
        
        # 按实验类型分组
        lines = ["# 实验结果对比\n"]
        
        # 消融实验
        ablation = df[df['exp_id'].str.startswith('E1')]
        if not ablation.empty:
            lines.append("## 消融实验\n")
            lines.append("| 实验 | 模态 | Val Acc | Val F1 |")
            lines.append("|------|------|---------|--------|")
            for _, row in ablation.iterrows():
                lines.append(f"| {row['exp_name']} | {row['modality']} | {row['val_acc']:.4f} | {row['val_f1']:.4f} |")
            lines.append("")
        
        # 融合策略对比
        fusion = df[df['exp_id'].str.startswith('E2')]
        if not fusion.empty:
            lines.append("## 融合策略对比\n")
            lines.append("| 融合方法 | Val Acc | Val F1 | Best Epoch |")
            lines.append("|----------|---------|--------|------------|")
            for _, row in fusion.iterrows():
                lines.append(f"| {row['fusion_type']} | {row['val_acc']:.4f} | {row['val_f1']:.4f} | {row['best_epoch']} |")
            lines.append("")
        
        # 数据增强对比
        aug = df[df['exp_id'].str.startswith('E3')]
        if not aug.empty:
            lines.append("## 数据增强对比\n")
            lines.append("| 增强策略 | Val Acc | Val F1 |")
            lines.append("|----------|---------|--------|")
            for _, row in aug.iterrows():
                lines.append(f"| {row['exp_name']} | {row['val_acc']:.4f} | {row['val_f1']:.4f} |")
            lines.append("")
        
        content = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ 对比表格已保存: {output_path}")
        
        return content


class TrainingHistoryRecorder:
    """训练过程记录器"""
    
    def __init__(self, exp_folder: str):
        self.exp_folder = exp_folder
        self.history = []
        self.current_epoch = {}
    
    def start_epoch(self, epoch: int):
        """开始新的epoch"""
        self.current_epoch = {'epoch': epoch}
    
    def log_train_metrics(self, loss: float, acc: float, f1: float):
        """记录训练指标"""
        self.current_epoch.update({
            'train_loss': loss,
            'train_acc': acc,
            'train_f1': f1
        })
    
    def log_val_metrics(self, loss: float, acc: float, f1: float, 
                        precision: float, recall: float):
        """记录验证指标"""
        self.current_epoch.update({
            'val_loss': loss,
            'val_acc': acc,
            'val_f1': f1,
            'val_precision': precision,
            'val_recall': recall
        })
    
    def log_lr(self, lr: float):
        """记录学习率"""
        self.current_epoch['learning_rate'] = lr
    
    def end_epoch(self, time_sec: float):
        """结束epoch"""
        self.current_epoch['time_sec'] = time_sec
        self.history.append(self.current_epoch.copy())
        self.current_epoch = {}
    
    def save(self):
        """保存训练历史"""
        history_path = os.path.join(self.exp_folder, 'training_history.csv')
        
        if not self.history:
            return
        
        fieldnames = list(self.history[0].keys())
        with open(history_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)


if __name__ == '__main__':
    # 测试
    logger = ExperimentLogger()
    logger.print_summary()
