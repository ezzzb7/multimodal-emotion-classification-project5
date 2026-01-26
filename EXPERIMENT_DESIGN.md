# 多模态情感分类 - 完整实验方案与技术路线

## 📊 数据集分析

### 当前数据规模
- **训练集**: 4000 样本 (含标签)
- **测试集**: 511 样本 (无标签，最终预测用)
- **类别分布**: positive(2388, 59.7%), negative(1193, 29.8%), neutral(419, 10.5%)

### ⚠️ 数据集划分原则（严格遵守）

```
原始数据 (train.txt, 4000样本)
    │
    ├── 训练集 (80%, 3200样本) ─── 用于模型训练、数据增强
    │                              ⚠️ Bad Case分析只能在训练集上做！
    │
    └── 验证集 (20%, 800样本) ─── 用于超参数调优、模型选择、Early Stopping
                                   ⚠️ 不能用于训练，不能用于数据增强！
    
测试集 (test_without_label.txt, 511样本)
    └── 只用于最终预测提交
        ⚠️ 整个实验过程中不能用于任何决策！
```

### 🚨 信息泄露警告

以下行为会导致**信息泄露**，使验证集结果虚高：

| 错误做法 | 正确做法 |
|---------|---------|
| ❌ 分析验证集Bad Case用于增强 | ✅ 只分析训练集Bad Case |
| ❌ 根据验证集结果手动调整数据 | ✅ 验证集只用于自动Early Stopping |
| ❌ 多次提交测试集选最高 | ✅ 测试集只预测一次 |
| ❌ 在验证集上训练 | ✅ 验证集只用于评估 |

### 推荐划分比例

| 划分方案 | 训练集 | 验证集 | 说明 |
|---------|-------|-------|------|
| 80/20 | 3200 | 800 | ✅ 推荐，验证集足够大 |
| 85/15 | 3400 | 600 | 当前使用 |
| 90/10 | 3600 | 400 | 验证集可能不够稳定 |

**建议**：使用80/20划分，验证集800样本能提供更稳定的评估。

---

## 🎯 实验设计思路

### 回答你的核心问题：是否需要先设计满意的模型？

**答案：不需要。** 正确的做法是：

1. **先确定实验框架和基线**
2. **系统地对比不同方法**
3. **每个实验都是有价值的数据点**
4. **最终选择最优方案**

实验的目的不是"找到最好的模型"，而是**通过对比实验证明你的设计选择是合理的**。

### 完整实验矩阵

```
┌─────────────────────────────────────────────────────────────────┐
│                     实验设计矩阵                                  │
├─────────────────────────────────────────────────────────────────┤
│ 维度1: 模态消融 (必须)                                           │
│   - Text-Only                                                    │
│   - Image-Only                                                   │
│   - Multimodal (融合)                                            │
├─────────────────────────────────────────────────────────────────┤
│ 维度2: 融合策略对比 (核心创新点)                                   │
│   - Late Fusion (concat)                                         │
│   - Early Fusion (element-wise)                                  │
│   - Cross-Attention                                              │
│   - [可选] Gated Fusion                                          │
│   - [可选] Tensor Fusion                                         │
├─────────────────────────────────────────────────────────────────┤
│ 维度3: 数据预处理对比                                            │
│   - 基础预处理 vs 增强预处理                                      │
│   - 文本清洗效果                                                  │
│   - 图像增强效果                                                  │
├─────────────────────────────────────────────────────────────────┤
│ 维度4: 编码器选择 (可选)                                         │
│   - DistilBERT vs BERT vs RoBERTa                               │
│   - ResNet50 vs ViT vs CLIP Image Encoder                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 推荐实验方案

### 阶段1: 基线建立与消融实验 (必须完成)

| 实验ID | 配置 | 目的 |
|-------|------|------|
| E1.1 | Late Fusion + 基础预处理 | 基线 |
| E1.2 | Text-Only | 消融：文本单模态 |
| E1.3 | Image-Only | 消融：图像单模态 |

### 阶段2: 融合策略对比 (核心实验)

| 实验ID | 配置 | 目的 |
|-------|------|------|
| E2.1 | Late Fusion | 对比基线 |
| E2.2 | Early Fusion | 特征级融合 |
| E2.3 | Cross-Attention | 注意力融合 |
| E2.4 | Gated Fusion | 门控融合 |

### 阶段3: 数据增强对比 (创新点)

| 实验ID | 配置 | 目的 |
|-------|------|------|
| E3.1 | 无增强 | 基线 |
| E3.2 | 文本增强 | 评估文本增强效果 |
| E3.3 | 图像增强 | 评估图像增强效果 |
| E3.4 | 双重增强 | 文本+图像同时增强 |

### 阶段4: 模型改进 (提升性能)

| 实验ID | 配置 | 目的 |
|-------|------|------|
| E4.1 | 使用CLIP特征 | 更强的预训练表示 |
| E4.2 | 集成多个模型 | 集成学习 |
| E4.3 | Bad Case针对性增强 | 数据驱动改进 |

---

## 📈 如何提升当前69%的准确率

### 分析当前瓶颈

当前最佳 Early Fusion 69%，可能的问题：
1. **数据量小** (3400样本) → 难以微调大模型
2. **类别不平衡** (neutral只有10%) → 分类器偏向positive
3. **预训练模型不够强** → 可尝试CLIP

### 改进方案优先级

| 优先级 | 方案 | 预期提升 | 复杂度 |
|-------|------|---------|-------|
| ⭐⭐⭐ | 模型集成 | +1-2% | 低 |
| ⭐⭐⭐ | 使用CLIP特征 | +2-5% | 中 |
| ⭐⭐ | 更强的数据增强 | +1-2% | 中 |
| ⭐⭐ | Focal Loss处理不平衡 | +0.5-1% | 低 |
| ⭐ | Label Smoothing | +0.5% | 低 |

### 🌟 推荐：CLIP方案

CLIP (Contrastive Language-Image Pre-training) 是当前多模态领域的SOTA：

```python
# CLIP的优势
1. 在4亿图文对上预训练，语义对齐能力强
2. 零样本能力强，小数据集也能工作
3. 图像和文本特征在同一语义空间
```

---

## 🛠️ 实验可复现性方案

### 1. 统一实验配置

```python
# configs/experiment_config.py
EXPERIMENT_SETTINGS = {
    # 固定超参数
    'seed': 42,
    'val_ratio': 0.2,  # 80/20划分
    'batch_size': 8,
    'accumulation_steps': 4,  # effective batch = 32
    'learning_rate': 2e-5,
    'num_epochs': 30,
    'early_stopping_patience': 5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    
    # 固定架构参数
    'feature_dim': 512,
    'dropout': 0.3,
    'freeze_encoders': True,
}
```

### 2. 实验结果记录

创建统一的CSV记录器：

```
experiments/
├── experiment_log.csv          # 所有实验汇总
├── E1.1_late_fusion/
│   ├── config.json             # 配置
│   ├── training_history.csv    # 训练历史
│   ├── best_model.pth          # 最佳模型
│   └── evaluation_results.json # 评估结果
├── E1.2_text_only/
│   └── ...
└── ...
```

### 3. 实验日志CSV格式

```csv
exp_id,exp_name,fusion_type,modality,val_acc,val_f1,train_acc,best_epoch,total_time,timestamp
E1.1,baseline_late,late,multimodal,0.6867,0.5706,0.7200,11,3925.68,2026-01-20
E1.2,text_only,none,text,0.6500,0.5422,0.6720,15,2913.37,2026-01-20
...
```

---

## 📚 参考论文与仓库

### 必读论文

1. **多模态融合综述**
   - Baltrušaitis et al. "Multimodal Machine Learning: A Survey and Taxonomy" (2019)
   - 涵盖Early/Late/Hybrid Fusion

2. **CLIP**
   - Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (2021)
   - OpenAI的图文对齐预训练

3. **BLIP**
   - Li et al. "BLIP: Bootstrapping Language-Image Pre-training" (2022)
   - Salesforce的多模态预训练

4. **注意力机制**
   - Vaswani et al. "Attention Is All You Need" (2017)
   - Transformer和Cross-Attention基础

5. **多模态情感分析**
   - Zadeh et al. "Multimodal Sentiment Intensity Analysis in Videos" (2016)
   - 经典多模态情感分析

### 参考仓库

1. **Hugging Face Transformers**
   - https://github.com/huggingface/transformers
   - BERT/RoBERTa/CLIP等模型

2. **OpenAI CLIP**
   - https://github.com/openai/CLIP
   - 官方CLIP实现

3. **PyTorch Image Models (timm)**
   - https://github.com/huggingface/pytorch-image-models
   - 图像编码器

4. **MultiBench**
   - https://github.com/pliang279/MultiBench
   - 多模态融合基准

5. **MMSA (Multimodal Sentiment Analysis)**
   - https://github.com/thuiar/MMSA
   - 多模态情感分析工具包

---

## ✅ 执行清单

### 立即执行

- [ ] 创建统一实验框架和配置
- [ ] 实现实验日志CSV记录器
- [ ] 修改数据划分为80/20
- [ ] 实现CLIP特征提取器

### 实验执行（按顺序）

- [ ] E1.1-E1.3: 基线和消融实验
- [ ] E2.1-E2.4: 融合策略对比
- [ ] E3.1-E3.4: 数据增强对比
- [ ] E4.1-E4.3: 模型改进

### 最终提交

- [ ] 生成所有实验结果对比表
- [ ] 绘制可视化图表
- [ ] 使用最佳模型预测测试集
- [ ] 整理代码和文档

---

## 🎯 预期最终结果

| 模型 | 预期准确率 | 说明 |
|------|----------|------|
| 基线 (Late Fusion) | ~68% | 参考点 |
| 最佳单模型 (CLIP-based) | ~72-75% | 使用CLIP特征 |
| 集成模型 | ~73-76% | 多模型投票 |

**合理预期**：通过系统实验，最终达到 **72-75%** 的验证集准确率是可行的。

