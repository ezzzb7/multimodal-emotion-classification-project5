"""
生成实验报告所需的可视化图表 - 精简版
逐个生成图表以避免内存问题
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

import matplotlib.pyplot as plt
import numpy as np
import os
import gc

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('figures', exist_ok=True)

def generate_fig1():
    """图1: 消融实验"""
    print("📊 生成图1: 消融实验对比...")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    experiments = ['Multimodal', 'Text-Only', 'Image-Only']
    accuracies = [67.00, 64.75, 62.62]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax.bar(experiments, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax.set_title('Ablation Study: Multimodal vs Single Modality', fontsize=12, fontweight='bold')
    ax.set_ylim(55, 72)
    
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.2f}%', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 标注提升
    ax.annotate('+2.25%', xy=(1, 65.5), fontsize=10, color='red', ha='center', fontweight='bold')
    ax.annotate('+4.38%', xy=(2, 63.5), fontsize=10, color='red', ha='center', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('figures/fig1_ablation_study.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ figures/fig1_ablation_study.png")
    gc.collect()

def generate_fig2():
    """图2: 融合方法对比"""
    print("📊 生成图2: 融合方法对比...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    methods = ['Late', 'Cross-Attn', 'Aligned', 'Hierarchical', 'Early', 'Gated']
    acc = [67.00, 66.75, 66.75, 64.75, 64.12, 61.00]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c', '#95a5a6']
    
    bars = ax.bar(methods, acc, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax.set_title('Comparison of Fusion Methods', fontsize=12, fontweight='bold')
    ax.set_ylim(55, 72)
    
    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('figures/fig2_fusion_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ figures/fig2_fusion_comparison.png")
    gc.collect()

def generate_fig3():
    """图3: 优化阶段提升"""
    print("📊 生成图3: 优化阶段提升...")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    stages = ['Baseline\n(Frozen)', 'Unfrozen\nEncoders', 'HP\nOptimization']
    acc = [67.00, 71.25, 72.25]
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    
    bars = ax.bar(stages, acc, color=colors, edgecolor='black', linewidth=1.5, width=0.5)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax.set_title('Optimization Progress: 67.00% → 72.25%', fontsize=12, fontweight='bold')
    ax.set_ylim(60, 78)
    
    for i, (bar, val) in enumerate(zip(bars, acc)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.2f}%', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        if i > 0:
            improvement = val - acc[i-1]
            ax.text(bar.get_x() + bar.get_width()/2, val - 2, f'+{improvement:.2f}%',
                   ha='center', fontsize=10, color='white', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('figures/fig3_optimization_progress.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ figures/fig3_optimization_progress.png")
    gc.collect()

def generate_fig4():
    """图4: 混淆矩阵"""
    print("📊 生成图4: 混淆矩阵...")
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 混淆矩阵 (行=真实, 列=预测)
    cm = np.array([
        [395, 18, 65],   # positive
        [45, 175, 18],   # negative  
        [61, 15, 8]      # neutral
    ])
    labels = ['Positive', 'Negative', 'Neutral']
    
    im = ax.imshow(cm, cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=10)
    
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title('Confusion Matrix (Val Acc: 72.25%)', fontsize=12, fontweight='bold')
    
    for i in range(3):
        for j in range(3):
            color = 'white' if cm[i, j] > 150 else 'black'
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', 
                   color=color, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('figures/fig4_confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ figures/fig4_confusion_matrix.png")
    gc.collect()

def generate_fig5():
    """图5: 数据预处理实验"""
    print("📊 生成图5: 数据预处理实验...")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    methods = ['Baseline', 'Image Aug', 'Text+Image', 'Text Clean']
    acc = [71.37, 71.13, 70.63, 70.00]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    bars = ax.bar(methods, acc, color=colors, edgecolor='black', linewidth=1.2, width=0.5)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax.set_title('Data Preprocessing Experiments', fontsize=12, fontweight='bold')
    ax.set_ylim(68, 73)
    
    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}%', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    bars[0].set_edgecolor('#27ae60')
    bars[0].set_linewidth(3)
    
    plt.tight_layout()
    fig.savefig('figures/fig5_data_preprocessing.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ figures/fig5_data_preprocessing.png")
    gc.collect()

def generate_fig6():
    """图6: 超参数搜索"""
    print("📊 生成图6: 超参数搜索...")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    configs = ['HP1\nDrop=0.2', 'HP4\nDrop=0.3', 'HP5\nDrop=0.25', 'HP2\nDrop=0.3', 'HP3\nDrop=0.3']
    acc = [72.25, 71.75, 71.63, 71.50, 70.63]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c']
    
    bars = ax.bar(configs, acc, color=colors, edgecolor='black', linewidth=1.2, width=0.5)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax.set_title('Hyperparameter Search Results', fontsize=12, fontweight='bold')
    ax.set_ylim(69, 74)
    
    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08, f'{val:.2f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    bars[0].set_edgecolor('#27ae60')
    bars[0].set_linewidth(3)
    
    plt.tight_layout()
    fig.savefig('figures/fig6_hyperparameter_search.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ figures/fig6_hyperparameter_search.png")
    gc.collect()

def generate_fig7():
    """图7: 错误类型分布"""
    print("📊 生成图7: 错误类型分布...")
    fig, ax = plt.subplots(figsize=(7, 6))
    
    error_types = ['pos to neu (65)', 'neu to pos (61)', 'neg to pos (45)', 
                   'neg to neu (18)', 'pos to neg (18)', 'neu to neg (15)']
    counts = [65, 61, 45, 18, 18, 15]
    colors = ['#e74c3c', '#c0392b', '#3498db', '#2980b9', '#f39c12', '#d35400']
    
    wedges, texts, autotexts = ax.pie(counts, labels=error_types, colors=colors, 
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 9})
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    ax.set_title('Error Type Distribution (222 errors)\nMain: pos/neu confusion (56.8%)', 
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig('figures/fig7_error_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ figures/fig7_error_distribution.png")
    gc.collect()

def generate_fig8():
    """图8: 类别分布"""
    print("📊 生成图8: 类别分布...")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [2388, 1193, 419]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    
    bars = ax.bar(labels, sizes, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Samples', fontsize=11)
    ax.set_title('Class Distribution (4000 samples)', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars, sizes):
        pct = val / 4000 * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, 
               f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.axhline(y=1333, color='gray', linestyle='--', alpha=0.7, label='Balanced (1333)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    fig.savefig('figures/fig8_class_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  ✓ figures/fig8_class_distribution.png")
    gc.collect()

if __name__ == '__main__':
    print("\n" + "="*50)
    print("开始生成实验报告图表...")
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
    print("✅ 所有图表已生成完成！")
    print("="*50)
    print("\n📁 生成的图表：")
    print("   1. fig1_ablation_study.png      - 消融实验 ⭐")
    print("   2. fig2_fusion_comparison.png   - 融合方法对比")
    print("   3. fig3_optimization_progress.png - 优化阶段提升 ⭐")
    print("   4. fig4_confusion_matrix.png    - 混淆矩阵 ⭐")
    print("   5. fig5_data_preprocessing.png  - 数据预处理实验")
    print("   6. fig6_hyperparameter_search.png - 超参数搜索")
    print("   7. fig7_error_distribution.png  - 错误类型分布")
    print("   8. fig8_class_distribution.png  - 类别分布 ⭐")
    print("\n⭐ 推荐放入实验报告的重要图表")
