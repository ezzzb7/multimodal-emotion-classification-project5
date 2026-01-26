"""
生成实验报告所需的可视化图表
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import seaborn as sns

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('figures', exist_ok=True)

# 设置风格
plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']

def save_fig(fig, name, dpi=100):
    """保存图片"""
    fig.savefig(f'figures/{name}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 已保存: figures/{name}.png")
    plt.close(fig)  # 立即释放内存

import gc  # 垃圾回收

# ============================================================
# 图1: 消融实验 - 验证多模态有效性
# ============================================================
print("\n📊 生成图1: 消融实验对比...")

fig, ax = plt.subplots(figsize=(8, 5))

experiments = ['Multimodal\n(A1)', 'Text-Only\n(A2)', 'Image-Only\n(A3)']
accuracies = [67.00, 64.75, 62.62]
f1_scores = [0.565, 0.533, 0.344]

x = np.arange(len(experiments))
width = 0.35

bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#3498db', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, [f*100 for f in f1_scores], width, label='F1 Score (×100)', color='#2ecc71', edgecolor='black', linewidth=1.2)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Ablation Study: Multimodal vs Single Modality', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(experiments, fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 80)

# 添加数值标签
for bar, val in zip(bars1, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.2f}%', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*100 + 1, f'{val:.3f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 添加提升标注
ax.annotate('', xy=(0, 67), xytext=(1, 64.75),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(0.5, 66.5, '+2.25%', ha='center', fontsize=10, color='red', fontweight='bold')

ax.annotate('', xy=(0, 67), xytext=(2, 62.62),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(1, 65, '+4.38%', ha='center', fontsize=10, color='red', fontweight='bold')

plt.tight_layout()
save_fig(fig, 'fig1_ablation_study')
gc.collect()

# ============================================================
# 图2: 融合方法对比
# ============================================================
print("📊 生成图2: 融合方法对比...")

fig, ax = plt.subplots(figsize=(10, 5))

fusion_methods = ['Late\nFusion', 'Cross-\nAttention', 'Aligned\nFusion', 'Hierarchical', 'Early\nFusion', 'Gated\nFusion']
fusion_acc = [67.00, 66.75, 66.75, 64.75, 64.12, 61.00]
fusion_colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c', '#95a5a6']

bars = ax.bar(fusion_methods, fusion_acc, color=fusion_colors, edgecolor='black', linewidth=1.2)

ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax.set_title('Comparison of Fusion Methods (Frozen Encoders)', fontsize=14, fontweight='bold')
ax.set_ylim(55, 72)

# 添加数值标签
for bar, val in zip(bars, fusion_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.2f}%', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 标注最佳
bars[0].set_edgecolor('#27ae60')
bars[0].set_linewidth(3)

ax.axhline(y=67.00, color='green', linestyle='--', alpha=0.5, label='Best: 67.00%')
ax.legend(loc='upper right')

plt.tight_layout()
save_fig(fig, 'fig2_fusion_comparison')
gc.collect()

# ============================================================
# 图3: 优化阶段提升图
# ============================================================
print("📊 生成图3: 优化阶段提升...")

fig, ax = plt.subplots(figsize=(9, 5))

stages = ['Baseline\n(Frozen Encoders)', 'Unfrozen Encoders\n+ Layer-wise LR', 'Hyperparameter\nOptimization']
stage_acc = [67.00, 71.25, 72.25]
stage_colors = ['#e74c3c', '#f39c12', '#2ecc71']

bars = ax.bar(stages, stage_acc, color=stage_colors, edgecolor='black', linewidth=1.5, width=0.6)

ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax.set_title('Optimization Progress: 67.00% → 72.25% (+5.25%)', fontsize=14, fontweight='bold')
ax.set_ylim(60, 78)

# 添加数值标签和提升
for i, (bar, val) in enumerate(zip(bars, stage_acc)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.2f}%', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    if i > 0:
        improvement = val - stage_acc[i-1]
        ax.annotate(f'+{improvement:.2f}%', 
                   xy=(bar.get_x() + bar.get_width()/2, val - 2),
                   fontsize=11, ha='center', color='white', fontweight='bold')

# 添加连接线
for i in range(len(stages)-1):
    ax.annotate('', xy=(i+1, stage_acc[i+1]-0.5), xytext=(i, stage_acc[i]+0.5),
                arrowprops=dict(arrowstyle='->', color='#34495e', lw=2))

plt.tight_layout()
save_fig(fig, 'fig3_optimization_progress')
gc.collect()

# ============================================================
# 图4: 混淆矩阵 (Bad Case 分析)
# ============================================================
print("📊 生成图4: 混淆矩阵...")

fig, ax = plt.subplots(figsize=(7, 6))

# 根据 Bad Case 分析结果构建混淆矩阵
# 验证集800样本，准确率72.25%，错误222个，正确578个
# 类别分布：positive 59.7% ≈ 478, negative 29.8% ≈ 238, neutral 10.5% ≈ 84

# 错误分布：
# positive → neutral: 65
# neutral → positive: 61  
# negative → positive: 45
# negative → neutral: 18
# positive → negative: 18
# neutral → negative: 15

# 推算混淆矩阵 (行=真实, 列=预测)
# positive: 478 - 65 - 18 = 395 correct
# negative: 238 - 45 - 18 = 175 correct
# neutral: 84 - 61 - 15 = 8 correct

confusion_matrix = np.array([
    [395, 18, 65],   # positive: 正确395, 预测为neg 18, 预测为neu 65
    [45, 175, 18],   # negative: 预测为pos 45, 正确175, 预测为neu 18
    [61, 15, 8]      # neutral: 预测为pos 61, 预测为neg 15, 正确8
])

labels = ['Positive', 'Negative', 'Neutral']

# 绘制热力图
im = ax.imshow(confusion_matrix, cmap='Blues')

# 添加颜色条
cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=11)

# 设置坐标轴
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix (HP1_BEST, Val Acc: 72.25%)', fontsize=13, fontweight='bold')

# 添加数值
for i in range(len(labels)):
    for j in range(len(labels)):
        val = confusion_matrix[i, j]
        color = 'white' if val > 200 else 'black'
        ax.text(j, i, f'{val}', ha='center', va='center', color=color, fontsize=14, fontweight='bold')

plt.tight_layout()
save_fig(fig, 'fig4_confusion_matrix')
gc.collect()

# ============================================================
# 图5: 数据预处理实验对比
# ============================================================
print("📊 生成图5: 数据预处理实验...")

fig, ax = plt.subplots(figsize=(9, 5))

preprocess_methods = ['DA1: Baseline\n(No Preprocessing)', 'DA3: Image\nAugmentation', 
                      'DA4: Text+Image', 'DA2: Text\nCleaning']
preprocess_acc = [71.37, 71.13, 70.63, 70.00]
preprocess_colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

bars = ax.bar(preprocess_methods, preprocess_acc, color=preprocess_colors, 
              edgecolor='black', linewidth=1.2, width=0.6)

ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax.set_title('Data Preprocessing Experiments', fontsize=14, fontweight='bold')
ax.set_ylim(68, 73)

for bar, val in zip(bars, preprocess_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}%', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# 标注最佳
bars[0].set_edgecolor('#27ae60')
bars[0].set_linewidth(3)
ax.axhline(y=71.37, color='green', linestyle='--', alpha=0.5)

plt.tight_layout()
save_fig(fig, 'fig5_data_preprocessing')
gc.collect()

# ============================================================
# 图6: 超参数搜索结果
# ============================================================
print("📊 生成图6: 超参数搜索...")

fig, ax = plt.subplots(figsize=(9, 5))

hp_configs = ['HP1\nDrop=0.2', 'HP4\nDrop=0.3', 'HP5\nDrop=0.25', 'HP2\nDrop=0.3', 'HP3\nDrop=0.3']
hp_acc = [72.25, 71.75, 71.63, 71.50, 70.63]
hp_colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c']

bars = ax.bar(hp_configs, hp_acc, color=hp_colors, edgecolor='black', linewidth=1.2, width=0.6)

ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax.set_title('Hyperparameter Search Results', fontsize=14, fontweight='bold')
ax.set_ylim(69, 74)

for bar, val in zip(bars, hp_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}%', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# 标注最佳
bars[0].set_edgecolor('#27ae60')
bars[0].set_linewidth(3)
ax.axhline(y=72.25, color='green', linestyle='--', alpha=0.5, label='Best: HP1 (72.25%)')
ax.legend(loc='upper right')

plt.tight_layout()
save_fig(fig, 'fig6_hyperparameter_search')
gc.collect()

# ============================================================
# 图7: 错误类型分布饼图
# ============================================================
print("📊 生成图7: 错误类型分布...")

fig, ax = plt.subplots(figsize=(8, 8))

error_types = ['pos→neu\n(65)', 'neu→pos\n(61)', 'neg→pos\n(45)', 
               'neg→neu\n(18)', 'pos→neg\n(18)', 'neu→neg\n(15)']
error_counts = [65, 61, 45, 18, 18, 15]
error_colors = ['#e74c3c', '#c0392b', '#3498db', '#2980b9', '#f39c12', '#d35400']

explode = (0.05, 0.05, 0, 0, 0, 0)  # 突出前两个最大的错误类型

wedges, texts, autotexts = ax.pie(error_counts, explode=explode, labels=error_types, 
                                   colors=error_colors, autopct='%1.1f%%',
                                   shadow=True, startangle=90,
                                   textprops={'fontsize': 11})

for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

ax.set_title('Error Type Distribution (222 errors total)\nMain Issue: pos↔neu confusion (56.8%)', 
             fontsize=13, fontweight='bold')

plt.tight_layout()
save_fig(fig, 'fig7_error_distribution')
gc.collect()

# ============================================================
# 图8: 模型架构示意图 (简化版)
# ============================================================
print("📊 生成图8: 模型架构...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# 绘制方框函数
def draw_box(ax, x, y, w, h, text, color='#3498db', fontsize=10):
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True)

# 输入层
draw_box(ax, 0.5, 8, 2.5, 1.2, 'Text Input', '#ecf0f1', 11)
draw_box(ax, 9, 8, 2.5, 1.2, 'Image Input', '#ecf0f1', 11)

# 编码器
draw_box(ax, 0.5, 5.5, 2.5, 1.8, 'DistilBERT\n(Unfrozen\nlast 2 layers)', '#3498db', 10)
draw_box(ax, 9, 5.5, 2.5, 1.8, 'ResNet50\n(Unfrozen\nlayer4)', '#e74c3c', 10)

# 投影层
draw_box(ax, 0.5, 4, 2.5, 1, '768 → 512', '#95a5a6', 10)
draw_box(ax, 9, 4, 2.5, 1, '2048 → 512', '#95a5a6', 10)

# Cross-Attention Fusion
draw_box(ax, 4, 2.5, 4, 1.5, 'Cross-Attention Fusion\nQ=Text, K/V=Image | Q=Image, K/V=Text', '#2ecc71', 10)

# 分类器
draw_box(ax, 4.5, 0.5, 3, 1.2, 'Classifier\n512 → 3', '#9b59b6', 11)

# 箭头
arrow_props = dict(arrowstyle='->', color='#34495e', lw=2)
ax.annotate('', xy=(1.75, 7.3), xytext=(1.75, 8), arrowprops=arrow_props)
ax.annotate('', xy=(10.25, 7.3), xytext=(10.25, 8), arrowprops=arrow_props)
ax.annotate('', xy=(1.75, 5), xytext=(1.75, 5.5), arrowprops=arrow_props)
ax.annotate('', xy=(10.25, 5), xytext=(10.25, 5.5), arrowprops=arrow_props)
ax.annotate('', xy=(4, 3.25), xytext=(3, 4.25), arrowprops=arrow_props)
ax.annotate('', xy=(8, 3.25), xytext=(9, 4.25), arrowprops=arrow_props)
ax.annotate('', xy=(6, 1.7), xytext=(6, 2.5), arrowprops=arrow_props)

# 学习率标注
ax.text(3.2, 6.3, 'LR: 1e-5', fontsize=9, color='#2980b9', style='italic')
ax.text(11.7, 6.3, 'LR: 1e-5', fontsize=9, color='#c0392b', style='italic')
ax.text(6, 2.1, 'LR: 5e-5', fontsize=9, color='#27ae60', style='italic')
ax.text(6, 0.2, 'LR: 1e-4', fontsize=9, color='#8e44ad', style='italic')

ax.set_title('Model Architecture: Cross-Attention Fusion with Layer-wise Learning Rates', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
save_fig(fig, 'fig8_model_architecture')
gc.collect()

# ============================================================
# 图9: 类别分布图
# ============================================================
print("📊 生成图9: 数据类别分布...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 训练集类别分布
labels = ['Positive', 'Negative', 'Neutral']
sizes = [2388, 1193, 419]
colors_pie = ['#2ecc71', '#e74c3c', '#f39c12']
explode = (0.02, 0.02, 0.05)

axes[0].pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[0].set_title('Training Data Distribution\n(4000 samples)', fontsize=13, fontweight='bold')

# 柱状图显示数量
bars = axes[1].bar(labels, sizes, color=colors_pie, edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('Number of Samples', fontsize=12)
axes[1].set_title('Class Imbalance Analysis', fontsize=13, fontweight='bold')

for bar, val in zip(bars, sizes):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, str(val), 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

# 添加不平衡比例标注
axes[1].axhline(y=1333, color='gray', linestyle='--', alpha=0.5, label='Balanced: 1333')
axes[1].legend(loc='upper right')

plt.tight_layout()
save_fig(fig, 'fig9_class_distribution')
gc.collect()

# ============================================================
# 汇总
# ============================================================
print("\n" + "="*60)
print("✅ 所有图表已生成完成！保存在 figures/ 目录")
print("="*60)
print("\n📁 生成的图表列表：")
print("   1. fig1_ablation_study      - 消融实验（证明多模态有效性）⭐")
print("   2. fig2_fusion_comparison   - 融合方法对比")
print("   3. fig3_optimization_progress - 优化阶段提升 ⭐")
print("   4. fig4_confusion_matrix    - 混淆矩阵（Bad Case分析）⭐")
print("   5. fig5_data_preprocessing  - 数据预处理实验")
print("   6. fig6_hyperparameter_search - 超参数搜索结果")
print("   7. fig7_error_distribution  - 错误类型分布")
print("   8. fig8_model_architecture  - 模型架构图 ⭐")
print("   9. fig9_class_distribution  - 数据类别分布")
print("\n⭐ 标记为推荐放入实验报告的重要图表")
print("\n📝 实验报告建议使用的图表：")
print("   - 图1 (消融实验): 证明多模态融合的有效性")
print("   - 图3 (优化提升): 展示从67%到72.25%的优化历程")  
print("   - 图4 (混淆矩阵): Bad Case分析，展示pos/neu混淆问题")
print("   - 图8 (模型架构): 清晰展示整体架构设计")
