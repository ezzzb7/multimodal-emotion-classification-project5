"""
CSV Logger - 保存训练历史到CSV文件，方便后续可视化
即使checkpoint删除也能保留训练曲线
"""
import os
import csv
from datetime import datetime


class CSVLogger:
    """保存训练历史到CSV文件"""
    
    def __init__(self, save_dir='training_logs', experiment_name=None):
        """
        Args:
            save_dir: CSV保存目录
            experiment_name: 实验名称
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.experiment_name = experiment_name
        self.csv_path = os.path.join(save_dir, f'{experiment_name}.csv')
        
        # 初始化CSV文件
        self._init_csv()
    
    def _init_csv(self):
        """初始化CSV文件，写入表头"""
        headers = [
            'epoch',
            'train_loss', 'train_acc', 'train_precision', 'train_recall', 'train_f1',
            'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1',
            'learning_rate', 'epoch_time'
        ]
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        print(f"✓ CSV Logger initialized: {self.csv_path}")
    
    def log_epoch(self, epoch, train_metrics, val_metrics, lr, epoch_time):
        """
        记录一个epoch的指标
        
        Args:
            epoch: 当前epoch
            train_metrics: 训练集指标字典
            val_metrics: 验证集指标字典
            lr: 当前学习率
            epoch_time: epoch耗时（秒）
        """
        row = [
            epoch,
            train_metrics.get('loss', 0),
            train_metrics.get('accuracy', 0),
            train_metrics.get('precision', 0),
            train_metrics.get('recall', 0),
            train_metrics.get('f1', 0),
            val_metrics.get('loss', 0),
            val_metrics.get('accuracy', 0),
            val_metrics.get('precision', 0),
            val_metrics.get('recall', 0),
            val_metrics.get('f1', 0),
            lr,
            epoch_time
        ]
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def save_config(self, config):
        """保存配置到单独的文件"""
        config_path = os.path.join(self.save_dir, f'{self.experiment_name}_config.txt')
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("="*50 + "\n")
            
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        
        print(f"✓ Config saved: {config_path}")


def create_visualization_script():
    """创建可视化脚本模板"""
    script_content = '''"""
训练历史可视化脚本
使用方法: python visualize_training.py <csv_file_path>
"""
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_training_history(csv_path):
    """绘制训练历史曲线"""
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training History - {os.path.basename(csv_path)}', fontsize=16)
    
    # 1. Loss曲线
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy曲线
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['train_acc'], label='Train Acc', marker='o')
    ax2.plot(df['epoch'], df['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.69, color='r', linestyle='--', label='Baseline (69%)')
    
    # 3. F1 Score曲线
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['train_f1'], label='Train F1', marker='o')
    ax3.plot(df['epoch'], df['val_f1'], label='Val F1', marker='s')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Learning Rate曲线
    ax4 = axes[1, 1]
    ax4.plot(df['epoch'], df['learning_rate'], label='Learning Rate', marker='o', color='green')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = csv_path.replace('.csv', '_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    
    # 显示图表
    plt.show()
    
    # 打印统计信息
    print("\\n" + "="*50)
    print("Training Statistics:")
    print("="*50)
    print(f"Best Val Accuracy: {df['val_acc'].max():.4f} at Epoch {df.loc[df['val_acc'].idxmax(), 'epoch']}")
    print(f"Best Val F1: {df['val_f1'].max():.4f} at Epoch {df.loc[df['val_f1'].idxmax(), 'epoch']}")
    print(f"Final Train Acc: {df['train_acc'].iloc[-1]:.4f}")
    print(f"Final Val Acc: {df['val_acc'].iloc[-1]:.4f}")
    print(f"Train-Val Gap: {(df['train_acc'].iloc[-1] - df['val_acc'].iloc[-1]):.4f}")
    print(f"Total Epochs: {len(df)}")
    print(f"Total Time: {df['epoch_time'].sum()/60:.2f} minutes")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_training.py <csv_file_path>")
        print("\\nAvailable CSV files:")
        for file in os.listdir('training_logs'):
            if file.endswith('.csv'):
                print(f"  - training_logs/{file}")
    else:
        csv_path = sys.argv[1]
        plot_training_history(csv_path)
'''
    
    with open('visualize_training.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✓ Visualization script created: visualize_training.py")


if __name__ == '__main__':
    # 创建可视化脚本
    create_visualization_script()
    print("\n使用方法:")
    print("1. 训练时会自动保存到 training_logs/*.csv")
    print("2. 可视化: python visualize_training.py training_logs/<experiment_name>.csv")
