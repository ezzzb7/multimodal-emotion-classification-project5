"""
画 train_acc vs val_acc 对比图，展示过拟合情况
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

os.makedirs('figures', exist_ok=True)

# 读取数据
df = pd.read_csv('experiments/all_results.csv')
print("实验数据：")
print(df[['exp_id', 'train_acc', 'val_acc', 'best_epoch']].to_string())

# 选择关键实验
key_exps = ['A1', 'A2', 'A3', 'F1', 'F2', 'F3', 'F4', 'OPT_late', 'OPT_cross_attention']
df_key = df[df['exp_id'].isin(key_exps)].copy()

# 计算过拟合差距
df_key['gap'] = df_key['train_acc'] - df_key['val_acc']

print("\n过拟合分析：")
print(df_key[['exp_id', 'train_acc', 'val_acc', 'gap']].to_string())

# 图1: Train vs Val Accuracy 对比
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(df_key))
width = 0.35

bars1 = ax.bar(x - width/2, df_key['train_acc'] * 100, width, label='Train Acc', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, df_key['val_acc'] * 100, width, label='Val Acc', color='#2ecc71', edgecolor='black')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Train vs Validation Accuracy (Overfitting Analysis)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_key['exp_id'], fontsize=10)
ax.legend(loc='upper left', fontsize=11)
ax.set_ylim(50, 100)

# 添加数值和过拟合标注
for i, (bar1, bar2, gap) in enumerate(zip(bars1, bars2, df_key['gap'])):
    # 训练准确率
    ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5, 
            f'{bar1.get_height():.1f}%', ha='center', va='bottom', fontsize=8, color='#2980b9')
    # 验证准确率
    ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5, 
            f'{bar2.get_height():.1f}%', ha='center', va='bottom', fontsize=8, color='#27ae60')
    
    # 过拟合严重的标红
    if gap > 0.15:
        ax.annotate(f'Gap: {gap*100:.1f}%', xy=(i, bar1.get_height() - 5), 
                   fontsize=9, ha='center', color='#e74c3c', fontweight='bold')

plt.tight_layout()
fig.savefig('figures/fig9_train_val_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("\n✓ 保存: figures/fig9_train_val_comparison.png")

# 图2: 过拟合差距排序图
fig, ax = plt.subplots(figsize=(10, 5))

df_sorted = df_key.sort_values('gap', ascending=True)
colors = ['#2ecc71' if g < 0.1 else '#f39c12' if g < 0.15 else '#e74c3c' for g in df_sorted['gap']]

bars = ax.barh(df_sorted['exp_id'], df_sorted['gap'] * 100, color=colors, edgecolor='black')
ax.set_xlabel('Overfitting Gap (Train Acc - Val Acc) %', fontsize=12)
ax.set_title('Overfitting Analysis by Experiment', fontsize=13, fontweight='bold')
ax.axvline(x=10, color='#f39c12', linestyle='--', alpha=0.7, label='Moderate (10%)')
ax.axvline(x=15, color='#e74c3c', linestyle='--', alpha=0.7, label='Severe (15%)')

for bar, gap in zip(bars, df_sorted['gap']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
            f'{gap*100:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')

ax.legend(loc='lower right')
plt.tight_layout()
fig.savefig('figures/fig10_overfitting_gap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("✓ 保存: figures/fig10_overfitting_gap.png")

print("\n完成！")
