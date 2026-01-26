# 高级融合方法设计文档

## 问题背景

老师指出的核心问题：
> "融合可以拼在一起加在一起，但是不够好，只能拿基础分。表征可能不在一个空间，怎么融合两个不同子空间？"

**根本原因**：
- 文本编码器(DistilBERT)输出的是"语言空间"的表征
- 图像编码器(ResNet)输出的是"视觉空间"的表征
- 这两个空间的语义完全不对齐，直接拼接/相加是在混合"苹果和橘子"

## 解决方案

### 方案对比

| 融合方法 | 是否解决空间对齐 | 复杂度 | 预期效果 |
|---------|-----------------|--------|---------|
| Late Fusion (拼接) | ❌ 否 | 低 | 基础分 |
| Early Fusion (相加) | ❌ 否 | 低 | 基础分 |
| Cross-Attention | ⚠️ 部分 | 中 | 中等 |
| **Aligned Fusion** | ✅ 是 | 高 | **高分** |
| **Hierarchical Fusion** | ✅ 是 | 高 | **高分** |

### 新增的高级融合方法

#### 1. Aligned Fusion (对齐融合) ⭐推荐

**核心思想**：先对齐，再融合

```
文本特征 ─→ [模态对齐层] ─→ 对齐文本 ─┐
                                    ├→ [跨模态Transformer] ─→ [多策略融合] ─→ 输出
图像特征 ─→ [模态对齐层] ─→ 对齐图像 ─┘
```

**三个关键组件**：

1. **模态对齐层 (Modality Alignment)**
   - 将文本和图像投影到共同的语义空间
   - 使用深层MLP + LayerNorm确保有效映射
   
2. **跨模态Transformer层 (Cross-Modal Transformer)**
   - 文本关注图像：学习"哪些图像区域与文本相关"
   - 图像关注文本：学习"哪些文本词汇描述了图像"
   - 双向交互，相互增强

3. **多策略融合**
   - 门控融合：动态学习模态权重
   - 双线性交互：捕捉复杂的跨模态关系
   - 残差连接：保留原始模态信息

**代码位置**: `models/advanced_fusion.py` -> `AlignedFusion`

#### 2. Hierarchical Fusion (层次化融合)

**核心思想**：模拟人类理解的层次性

```
Level 1 (低层): 特征提取和初步融合
     ↓
Level 2 (高层): 语义理解和跨模态注意力
     ↓
Level 3 (决策): 整合所有层次信息做最终决策
```

**设计原理**：
- 低层融合捕捉浅层模式（如：图像颜色与文本情感词）
- 高层融合捕捉深层语义（如：图像场景与文本主题）
- 决策融合综合所有层次的信息

**代码位置**: `models/advanced_fusion.py` -> `HierarchicalFusion`

## 实验设计

### 完整实验计划

```
Phase 1: 消融实验 (Ablation Study)
├── A1: Multimodal (Late Fusion) - 基线
├── A2: Text Only - 证明图像的价值
└── A3: Image Only - 证明文本的价值

Phase 2: 基础融合对比 (Basic Fusion)
├── F1: Late Fusion - 简单拼接
├── F2: Early Fusion - 简单相加
├── F3: Cross-Attention - 注意力机制
└── F4: Gated Fusion - 门控权重

Phase 3: 高级融合 (Advanced Fusion) ⭐核心实验
├── AF1: Aligned Fusion - 模态对齐 + 跨模态Transformer
└── AF2: Hierarchical Fusion - 层次化多级融合
```

### 预期结果

| 方法 | 预期准确率 | 说明 |
|-----|-----------|------|
| Text Only | ~65% | 文本主导情感 |
| Image Only | ~60% | 图像辅助判断 |
| Late Fusion | ~68% | 基础多模态 |
| **Aligned Fusion** | **~72%** | 解决空间对齐问题 |
| **Hierarchical** | **~71%** | 多层次融合 |

## 运行命令

```bash
# 删除旧的数据划分，确保使用80/20
rm -rf splits/

# Phase 1: 消融实验
python run_experiment_simple.py --phase 1

# Phase 2: 基础融合对比
python run_experiment_simple.py --phase 2

# Phase 3: 高级融合（核心实验）
python run_experiment_simple.py --phase 3

# 查看所有结果
python run_experiment_simple.py --results
```

## 报告撰写要点

### 1. 问题引出
> 简单的特征拼接(concatenation)或逐元素相加(element-wise addition)虽然实现简单，
> 但忽略了一个关键问题：不同模态的表征处于不同的语义空间，直接组合无法实现有效的信息融合。

### 2. 解决方案
> 本文提出Aligned Fusion方法，通过三个核心模块解决空间对齐问题：
> 1. 模态对齐层：将异构表征投影到共同语义空间
> 2. 跨模态Transformer：实现双向跨模态信息交互
> 3. 多策略融合：综合门控、双线性交互等多种融合信号

### 3. 引用论文
- Attention机制: Vaswani et al. "Attention is All You Need" (2017)
- 多模态融合: Baltrusaitis et al. "Multimodal Machine Learning: A Survey" (2019)
- 对比对齐思想: Radford et al. "Learning Transferable Visual Models" (CLIP, 2021)

## 文件结构

```
models/
├── advanced_fusion.py    # 新增：高级融合方法
│   ├── ModalityAlignmentLayer
│   ├── CrossModalTransformerLayer
│   ├── AlignedFusion
│   ├── ContrastiveAlignedFusion
│   └── HierarchicalFusion
├── fusion.py             # 原有：基础融合方法
└── multimodal_model.py   # 已修改：支持新融合方法
```
