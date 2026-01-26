"""
生成实验报告图表 - 修复文字重叠问题
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import gc

# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

os.makedirs('figures', exist_ok=True)

def generate_fig1():
    """图1: 消融实验"""
    print("📊 生成图1: 消融实验...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    experiments = ['Multimodal', 'Text-Only', 'Image-Only']
    accuracies = [67.00, 64.75, 62.62]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax.bar(experiments, accuracies, color=colors, edgecolor='black', linewidth=1.5, width=0.6)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Ablation Study: Multimodal vs Single Modality', fontsize=13, fontweight='bold')
    ax.set_ylim(55, 75)
    
    # 数值标签 - 放在柱子上方
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
    
    # 提升标注 - 用箭头和独立文本，避免重叠
    ax.annotate('', xy=(0, 67), xytext=(1, 67),
                arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=2))
    ax.text(0.5, 68.5, '+2.25%', ha='center', fontsize=11, color='#e74c3c', fontweight='bold')
    
    ax.annotate('', xy=(0, 67), xytext=(2, 67),
                arrowprops=dict(arrowstyle='<->', color='#e74c3c', lw=2))
    ax.text(1, 70, '+4.38%', ha='center', fontsize=11, color='#e74c3c', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('figures/fig1_ablation_study.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ figures/fig1_ablation_study.png")
    gc.collect()

def generate_fig2():
    """图2: 融合方法对比"""
    print("📊 生成图2: 融合方法对比...")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    methods = ['Late\nFusion', 'Cross-\nAttention', 'Aligned\nFusion', 'Hierarchical', 'Early\nFusion', 'Gated\nFusion']
    acc = [67.00, 66.75, 66.75, 64.75, 64.12, 61.00]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c', '#95a5a6']
    
    bars = ax.bar(methods, acc, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Comparison of Fusion Methods (Frozen Encoders)', fontsize=13, fontweight='bold')
    ax.set_ylim(55, 72)
    
    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
    
    # 标注最佳
    ax.axhline(y=67.00, color='#27ae60', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(5.5, 67.3, 'Best', fontsize=10, color='#27ae60', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('figures/fig2_fusion_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ figures/fig2_fusion_comparison.png")
    gc.collect()

def generate_fig3():
    """图3: 优化阶段提升"""
    print("📊 生成图3: 优化阶段提升...")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    stages = ['Baseline\n(Frozen Encoders)', 'Unfrozen Encoders\n+ Layer-wise LR', 'Hyperparameter\nOptimization']
    acc = [67.00, 71.25, 72.25]
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    
    bars = ax.bar(stages, acc, color=colors, edgecolor='black', linewidth=1.5, width=0.5)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Optimization Progress: 67.00% → 72.25% (+5.25%)', fontsize=13, fontweight='bold')
    ax.set_ylim(60, 78)
    
    # 数值标签放在柱子顶部
    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, 
                f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
    
    # 提升标注 - 放在柱子之间，用不同颜色
    ax.annotate('+4.25%', xy=(0.5, 69), fontsize=11, ha='center', 
                color='#d35400', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaa7', edgecolor='#d35400'))
    ax.annotate('+1.00%', xy=(1.5, 71.8), fontsize=11, ha='center', 
                color='#27ae60', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', edgecolor='#27ae60'))
    
    # 连接箭头
    ax.annotate('', xy=(1, 71.25), xytext=(0, 67),
                arrowprops=dict(arrowstyle='->', color='#d35400', lw=2))
    ax.annotate('', xy=(2, 72.25), xytext=(1, 71.25),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
    
    plt.tight_layout()
    fig.savefig('figures/fig3_optimization_progress.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ figures/fig3_optimization_progress.png")
    gc.collect()

def generate_fig4():
    """图4: 混淆矩阵"""
    print("📊 生成图4: 混淆矩阵...")
    fig, ax = plt.subplots(figsize=(7, 6))
    
    cm = np.array([
        [395, 18, 65],
        [45, 175, 18],
        [61, 15, 8]
    ])
    labels = ['Positive', 'Negative', 'Neutral']
    
    im = ax.imshow(cm, cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=11)
    
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix (HP1_BEST, Val Acc: 72.25%)', fontsize=12, fontweight='bold')
    
    # 数值标签
    for i in range(3):
        for j in range(3):
            val = cm[i, j]
            color = 'white' if val > 150 else 'black'
            ax.text(j, i, f'{val}', ha='center', va='center', 
                   color=color, fontsize=14, fontweight='bold')
    
    # 标注主要错误区域
    rect1 = plt.Rectangle((1.5, -0.5), 1, 1, fill=False, edgecolor='red', linewidth=3)
    rect2 = plt.Rectangle((-0.5, 1.5), 1, 1, fill=False, edgecolor='red', linewidth=3)
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    plt.tight_layout()
    fig.savefig('figures/fig4_confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ figures/fig4_confusion_matrix.png")
    gc.collect()

def generate_fig5():
    """图5: 数据预处理实验"""
    print("📊 生成图5: 数据预处理实验...")
    fig, ax = plt.subplots(figsize=(9, 5))
    
    methods = ['Baseline\n(No Preprocessing)', 'Image\nAugmentation', 'Text+Image\nPreprocessing', 'Text\nCleaning']
    acc = [71.37, 71.13, 70.63, 70.00]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    bars = ax.bar(methods, acc, color=colors, edgecolor='black', linewidth=1.2, width=0.55)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Data Preprocessing Experiments', fontsize=13, fontweight='bold')
    ax.set_ylim(68, 73)
    
    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, 
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
    
    # 标注最佳
    bars[0].set_edgecolor('#27ae60')
    bars[0].set_linewidth(3)
    ax.axhline(y=71.37, color='#27ae60', linestyle='--', alpha=0.7)
    ax.text(3.5, 71.5, 'Best: No Preprocessing', fontsize=10, color='#27ae60', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('figures/fig5_data_preprocessing.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ figures/fig5_data_preprocessing.png")
    gc.collect()

def generate_fig6():
    """图6: 超参数搜索"""
    print("📊 生成图6: 超参数搜索...")
    fig, ax = plt.subplots(figsize=(9, 5))
    
    configs = ['HP1\nDrop=0.2\nLR=1e-4', 'HP4\nDrop=0.3\nLR=2e-4', 'HP5\nDrop=0.25\nLR=1e-4', 
               'HP2\nDrop=0.3\nLR=1e-4', 'HP3\nDrop=0.3\nLR=5e-5']
    acc = [72.25, 71.75, 71.63, 71.50, 70.63]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c']
    
    bars = ax.bar(configs, acc, color=colors, edgecolor='black', linewidth=1.2, width=0.55)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Hyperparameter Search Results', fontsize=13, fontweight='bold')
    ax.set_ylim(69, 74)
    
    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
    
    bars[0].set_edgecolor('#27ae60')
    bars[0].set_linewidth(3)
    ax.axhline(y=72.25, color='#27ae60', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.savefig('figures/fig6_hyperparameter_search.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ figures/fig6_hyperparameter_search.png")
    gc.collect()

def generate_fig7():
    """图7: 错误类型分布"""
    print("📊 生成图7: 错误类型分布...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 使用水平条形图避免标签重叠
    error_types = ['pos to neu', 'neu to pos', 'neg to pos', 'neg to neu', 'pos to neg', 'neu to neg']
    counts = [65, 61, 45, 18, 18, 15]
    colors = ['#e74c3c', '#c0392b', '#3498db', '#2980b9', '#f39c12', '#d35400']
    
    bars = ax.barh(error_types, counts, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Error Count', fontsize=12)
    ax.set_title('Error Type Distribution (222 errors total)', fontsize=13, fontweight='bold')
    
    for bar, val in zip(bars, counts):
        pct = val / 222 * 100
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{val} ({pct:.1f}%)', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 标注主要问题
    ax.axvline(x=60, color='red', linestyle='--', alpha=0.5)
    ax.text(62, 5.5, 'Main Issue:\npos/neu confusion\n(56.8%)', fontsize=10, color='#c0392b', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('figures/fig7_error_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ figures/fig7_error_distribution.png")
    gc.collect()

def generate_fig8():
    """图8: 类别分布"""
    print("📊 生成图8: 类别分布...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [2388, 1193, 419]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    
    bars = ax.bar(labels, sizes, color=colors, edgecolor='black', linewidth=1.5, width=0.6)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution in Training Data (4000 samples)', fontsize=13, fontweight='bold')
    
    for bar, val in zip(bars, sizes):
        pct = val / 4000 * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
               f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
    
    ax.axhline(y=1333, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(2.3, 1400, 'Balanced: 1333', fontsize=10, color='gray', fontweight='bold')
    
    # 标注不平衡问题
    ax.annotate('Severely\nImbalanced!', xy=(2, 419), xytext=(2.3, 800),
                fontsize=10, color='#d35400', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#d35400', lw=2))
    
    plt.tight_layout()
    fig.savefig('figures/fig8_class_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ figures/fig8_class_distribution.png")
    gc.collect()

if __name__ == '__main__':
    print("\n" + "="*50)
    print("重新生成实验报告图表（修复文字重叠）...")
    print("="*50 + "\n")
    
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig6()
    generate_fig7()
    generate_fig8()
    
    print("\n" + "="*50)
    print("✅ 所有图表已重新生成！")
    print("="*50)
