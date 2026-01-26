"""
实验结果可视化分析
生成训练曲线、对比图表、混淆矩阵等
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ExperimentVisualizer:
    """实验结果可视化器"""
    
    def __init__(self, experiment_dir: str = 'experiments'):
        self.experiment_dir = experiment_dir
        self.output_dir = os.path.join(experiment_dir, 'visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_training_curves(self, exp_folder: str):
        """绘制单个实验的训练曲线"""
        history_path = os.path.join(exp_folder, 'training_history.csv')
        
        if not os.path.exists(history_path):
            print(f"⚠️ 未找到训练历史: {history_path}")
            return
        
        df = pd.read_csv(history_path)
        exp_name = os.path.basename(exp_folder)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Curves - {exp_name}', fontsize=14)
        
        # Loss
        axes[0, 0].plot(df['epoch'], df['train_loss'], 'b-o', label='Train', markersize=4)
        axes[0, 0].plot(df['epoch'], df['val_loss'], 'r-s', label='Val', markersize=4)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(df['epoch'], df['train_acc'], 'b-o', label='Train', markersize=4)
        axes[0, 1].plot(df['epoch'], df['val_acc'], 'r-s', label='Val', markersize=4)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 0].plot(df['epoch'], df['train_f1'], 'b-o', label='Train', markersize=4)
        axes[1, 0].plot(df['epoch'], df['val_f1'], 'r-s', label='Val', markersize=4)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        if 'learning_rate' in df.columns:
            axes[1, 1].plot(df['epoch'], df['learning_rate'], 'g-o', markersize=4)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'{exp_name}_curves.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 训练曲线已保存: {output_path}")
    
    def plot_comparison_bar(self, metric: str = 'val_acc'):
        """绘制实验对比柱状图"""
        summary_path = os.path.join(self.experiment_dir, 'experiment_summary.csv')
        
        if not os.path.exists(summary_path):
            print("⚠️ 未找到实验汇总文件")
            return
        
        df = pd.read_csv(summary_path)
        
        if len(df) == 0:
            print("⚠️ 没有实验数据")
            return
        
        # 按实验ID排序
        df = df.sort_values('exp_id')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 颜色映射
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
        
        bars = ax.bar(df['exp_name'], df[metric], color=colors)
        
        # 添加数值标签
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Experiment')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Experiment Comparison - {metric.replace("_", " ").title()}')
        ax.set_xticklabels(df['exp_name'], rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        
        # 添加基线
        if metric == 'val_acc':
            ax.axhline(y=0.69, color='red', linestyle='--', label='Baseline (69%)')
            ax.legend()
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'comparison_{metric}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 对比图已保存: {output_path}")
    
    def plot_ablation_study(self):
        """绘制消融实验对比"""
        summary_path = os.path.join(self.experiment_dir, 'experiment_summary.csv')
        
        if not os.path.exists(summary_path):
            return
        
        df = pd.read_csv(summary_path)
        ablation = df[df['exp_id'].str.startswith('E1')]
        
        if len(ablation) == 0:
            print("⚠️ 未找到消融实验数据")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(ablation))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, ablation['val_acc'], width, label='Accuracy', color='steelblue')
        bars2 = ax.bar(x + width/2, ablation['val_f1'], width, label='F1 Score', color='coral')
        
        ax.set_ylabel('Score')
        ax.set_title('Ablation Study: Multimodal vs Single Modal')
        ax.set_xticks(x)
        ax.set_xticklabels(ablation['modality'])
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'ablation_study.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 消融实验图已保存: {output_path}")
    
    def plot_fusion_comparison(self):
        """绘制融合策略对比"""
        summary_path = os.path.join(self.experiment_dir, 'experiment_summary.csv')
        
        if not os.path.exists(summary_path):
            return
        
        df = pd.read_csv(summary_path)
        fusion = df[df['exp_id'].str.startswith('E2')]
        
        if len(fusion) == 0:
            print("⚠️ 未找到融合对比实验数据")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
        bars = ax.barh(fusion['fusion_type'], fusion['val_acc'], color=colors[:len(fusion)])
        
        ax.set_xlabel('Validation Accuracy')
        ax.set_title('Fusion Strategy Comparison')
        ax.grid(True, axis='x', alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{bar.get_width():.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'fusion_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 融合对比图已保存: {output_path}")
    
    def plot_all_training_curves(self):
        """为所有实验绘制训练曲线"""
        for folder in os.listdir(self.experiment_dir):
            folder_path = os.path.join(self.experiment_dir, folder)
            if os.path.isdir(folder_path) and folder != 'visualizations':
                self.plot_training_curves(folder_path)
    
    def generate_all_visualizations(self):
        """生成所有可视化"""
        print("\n" + "="*50)
        print("生成可视化图表")
        print("="*50)
        
        # 训练曲线
        self.plot_all_training_curves()
        
        # 对比图
        self.plot_comparison_bar('val_acc')
        self.plot_comparison_bar('val_f1')
        
        # 消融实验
        self.plot_ablation_study()
        
        # 融合对比
        self.plot_fusion_comparison()
        
        print(f"\n✓ 所有可视化已保存到: {self.output_dir}")
    
    def generate_report_figures(self):
        """生成适合报告使用的高质量图表"""
        print("\n生成报告图表...")
        
        summary_path = os.path.join(self.experiment_dir, 'experiment_summary.csv')
        if not os.path.exists(summary_path):
            print("⚠️ 未找到实验数据")
            return
        
        df = pd.read_csv(summary_path)
        
        # 综合对比图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：准确率对比
        ax1 = axes[0]
        colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
        bars = ax1.barh(df['exp_name'], df['val_acc'], color=colors)
        ax1.set_xlabel('Validation Accuracy')
        ax1.set_title('(a) Model Accuracy Comparison')
        ax1.axvline(x=0.69, color='red', linestyle='--', linewidth=1, label='Baseline')
        ax1.legend()
        for bar in bars:
            ax1.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                    f'{bar.get_width():.3f}', va='center', fontsize=8)
        
        # 右图：多指标对比
        ax2 = axes[1]
        metrics = ['val_acc', 'val_f1', 'val_precision', 'val_recall']
        x = np.arange(len(df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                ax2.bar(x + i*width, df[metric], width, label=metric.replace('val_', ''))
        
        ax2.set_ylabel('Score')
        ax2.set_title('(b) Multi-metric Comparison')
        ax2.set_xticks(x + width*1.5)
        ax2.set_xticklabels(df['exp_id'], rotation=45)
        ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'report_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ 报告图表已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='实验结果可视化')
    parser.add_argument('--exp-dir', type=str, default='experiments',
                        help='实验目录')
    parser.add_argument('--exp', type=str, default=None,
                        help='指定实验文件夹（绘制单个实验曲线）')
    parser.add_argument('--all', action='store_true',
                        help='生成所有可视化')
    parser.add_argument('--report', action='store_true',
                        help='生成报告用图表')
    
    args = parser.parse_args()
    
    visualizer = ExperimentVisualizer(args.exp_dir)
    
    if args.exp:
        visualizer.plot_training_curves(args.exp)
    elif args.all:
        visualizer.generate_all_visualizations()
    elif args.report:
        visualizer.generate_report_figures()
    else:
        visualizer.generate_all_visualizations()


if __name__ == '__main__':
    main()
