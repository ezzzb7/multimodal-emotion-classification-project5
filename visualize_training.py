"""
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
    print("\n" + "="*50)
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
        print("\nAvailable CSV files:")
        for file in os.listdir('training_logs'):
            if file.endswith('.csv'):
                print(f"  - training_logs/{file}")
    else:
        csv_path = sys.argv[1]
        plot_training_history(csv_path)
