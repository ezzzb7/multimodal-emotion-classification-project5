"""
Visualization utilities for training results
用于生成论文和报告所需的各种可视化图表
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List


# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')


def plot_training_curves(history: Dict, save_path: str = None):
    """
    绘制训练曲线（Loss和Accuracy）
    
    Args:
        history: 训练历史字典（从logger.get_history()获取）
        save_path: 保存路径（如果为None则显示）
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 curves
    axes[1, 0].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    axes[1, 0].plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('F1 Score', fontsize=12)
    axes[1, 0].set_title('F1 Score Curves', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 1].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(experiments_file: str, save_path: str = None):
    """
    绘制多个实验的指标对比图（用于消融实验）
    
    Args:
        experiments_file: 实验对比CSV文件路径
        save_path: 保存路径
    """
    # 读取数据
    df = pd.read_csv(experiments_file)
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准备数据
    experiments = df['experiment_name'].tolist()
    val_acc = df['best_val_acc'].tolist()
    val_f1 = df['best_val_f1'].tolist()
    
    x = np.arange(len(experiments))
    width = 0.35
    
    # Accuracy comparison
    axes[0].bar(x, val_acc, width, label='Validation Accuracy', color='steelblue')
    axes[0].set_xlabel('Experiment', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Comparison - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(experiments, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, v in enumerate(val_acc):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    # F1 comparison
    axes[1].bar(x, val_f1, width, label='Validation F1', color='coral')
    axes[1].set_xlabel('Experiment', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('Model Comparison - F1 Score', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(experiments, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, v in enumerate(val_f1):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Metrics comparison saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix_comparison(cm_files: Dict[str, str], save_path: str = None):
    """
    并排绘制多个混淆矩阵（用于对比不同模型）
    
    Args:
        cm_files: 混淆矩阵文件字典 {model_name: csv_path}
        save_path: 保存路径
    """
    n_models = len(cm_files)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    class_names = ['Positive', 'Negative', 'Neutral']
    
    for ax, (model_name, cm_file) in zip(axes, cm_files.items()):
        # 读取混淆矩阵
        cm = pd.read_csv(cm_file, index_col=0).values
        
        # 归一化
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 绘制热图
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, cbar_kws={'label': 'Normalized Count'})
        
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix comparison saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_results_table(experiments_file: str, save_path: str = None):
    """
    创建实验结果对比表格（LaTeX格式，用于论文）
    
    Args:
        experiments_file: 实验对比CSV文件路径
        save_path: 保存路径（.tex文件）
    """
    df = pd.read_csv(experiments_file)
    
    # 选择关键列
    df_display = df[['experiment_name', 'fusion_type', 'best_val_acc', 'best_val_f1']]
    
    # 重命名列
    df_display.columns = ['Model', 'Fusion Type', 'Accuracy', 'F1 Score']
    
    # 格式化数值
    df_display['Accuracy'] = df_display['Accuracy'].apply(lambda x: f'{x:.4f}')
    df_display['F1 Score'] = df_display['F1 Score'].apply(lambda x: f'{x:.4f}')
    
    # 生成LaTeX表格
    latex_table = df_display.to_latex(index=False, escape=False)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        print(f"✓ LaTeX table saved: {save_path}")
    
    # 同时保存为Markdown格式
    if save_path:
        md_path = save_path.replace('.tex', '.md')
        md_table = df_display.to_markdown(index=False)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_table)
        print(f"✓ Markdown table saved: {md_path}")
    
    return df_display


def plot_error_analysis(error_samples_file: str, save_path: str = None):
    """
    分析预测错误的样本分布
    
    Args:
        error_samples_file: 错误样本CSV文件路径
        save_path: 保存路径
    """
    df = pd.read_csv(error_samples_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 真实标签分布
    true_label_counts = df['true_label'].value_counts().sort_index()
    axes[0].bar(true_label_counts.index, true_label_counts.values, color='steelblue')
    axes[0].set_xlabel('True Label', fontsize=12)
    axes[0].set_ylabel('Error Count', fontsize=12)
    axes[0].set_title('Error Distribution by True Label', fontsize=14, fontweight='bold')
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(['Positive', 'Negative', 'Neutral'])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 预测标签分布
    pred_label_counts = df['pred_label'].value_counts().sort_index()
    axes[1].bar(pred_label_counts.index, pred_label_counts.values, color='coral')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('Error Count', fontsize=12)
    axes[1].set_title('Error Distribution by Predicted Label', fontsize=14, fontweight='bold')
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(['Positive', 'Negative', 'Neutral'])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Error analysis saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_all_visualizations(log_dir: str, output_dir: str = None):
    """
    生成所有可视化图表（一键生成）
    
    Args:
        log_dir: 日志目录
        output_dir: 输出目录（如果为None则保存在log_dir）
    """
    if output_dir is None:
        output_dir = os.path.join(log_dir, 'visualizations')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}\n")
    
    # 1. 训练曲线
    history_file = os.path.join(log_dir, 'history.json')
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            data = json.load(f)
            history = data['history']
        
        curves_path = os.path.join(output_dir, 'training_curves.png')
        plot_training_curves(history, curves_path)
    
    # 2. 实验对比
    comparison_file = os.path.join(os.path.dirname(log_dir), 'experiments_comparison.csv')
    if os.path.exists(comparison_file):
        comparison_path = os.path.join(output_dir, 'metrics_comparison.png')
        plot_metrics_comparison(comparison_file, comparison_path)
        
        # 生成结果表格
        table_path = os.path.join(output_dir, 'results_table.tex')
        create_results_table(comparison_file, table_path)
    
    # 3. 错误分析
    error_file = os.path.join(log_dir, 'error_samples.csv')
    if os.path.exists(error_file):
        error_path = os.path.join(output_dir, 'error_analysis.png')
        plot_error_analysis(error_file, error_path)
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # 示例用法
    import sys
    
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
        generate_all_visualizations(log_dir)
    else:
        print("Usage: python visualize.py <log_dir>")
