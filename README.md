# 多模态情感分类 (Multimodal Sentiment Classification)

**GitHub**: https://github.com/ezzzb7/multimodal-emotion-classification-project5

基于文本和图像的多模态情感分类系统，实现了6种融合策略，包含完整的消融实验、数据预处理实验、超参数搜索和Bad Case分析。

**最佳验证集准确率：72.25%**

---

##  项目简介

### 任务描述

给定配对的文本和图像，预测对应的情感标签（三分类任务）：
- **positive**：正面情感
- **neutral**：中性情感
- **negative**：负面情感

### 主要特性

- **6种融合策略**：Late Fusion、Early Fusion、Cross-Attention、Gated Fusion、Aligned Fusion、Hierarchical Fusion
- **完整消融实验**：Text-Only、Image-Only、Multimodal对比
- **数据预处理实验**：文本清洗、图像增强效果验证
- **超参数搜索**：Dropout、学习率、权重衰减调优
- **Bad Case分析**：错误样本分析与针对性改进
- **断点续传**：支持训练中断后继续训练
- **分层学习率**：预训练层、融合层、分类层使用不同学习率

### 最佳性能

| 配置 | 验证集准确率 | 验证集F1 |
|------|-------------|----------|
| **HP1_BEST (Cross-Attention + 解冻编码器)** | **72.25%** | 0.5940 |

---

### 环境配置

```bash
# 1. 克隆仓库
git clone https://github.com/ezzzb7/multimodal-emotion-classification-project5.git
cd multimodal-emotion-classification-project5

# 2. 创建虚拟环境
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 预训练模型下载

本项目使用的预训练模型：
- **DistilBERT**: `distilbert-base-uncased`
- **ResNet50**: `torchvision.models.resnet50(pretrained=True)`

首次运行时会自动下载，也可以提前下载：

```python
from transformers import DistilBertModel, DistilBertTokenizer
DistilBertModel.from_pretrained('distilbert-base-uncased')
DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
```

---

##  数据准备

### 数据集结构

**注意：数据文件夹需放在代码文件夹的父目录中**

```
D:\当代人工智能\project5\          # 项目根目录
├── data/                          # 数据文件夹（包含所有图像和文本）
│   ├── 1.txt                      # 文本文件
│   ├── 1.jpg                      # 对应图像文件
│   ├── 2.txt
│   ├── 2.jpg
│   └── ...                        # 共4000个训练样本 + 511个测试样本
├── train.txt                      # 训练标签文件
├── test_without_label.txt         # 测试文件（标签为null）
└── code/                          # 代码文件夹（本仓库）
    ├── README.md
    ├── requirements.txt
    └── ...
```
---

##  代码结构

```
code/
├──  configs/              # 配置模块
│   ├── config.py           # 基础训练配置（batch size、学习率、epoch等）
│   └── experiment_config.py # 实验配置（各阶段实验参数）
│
├──  data/                 # 数据处理模块
│   ├── data_loader.py      # 数据加载器
│   ├── dataset.py          # Dataset类
│   ├── preprocessing.py    # 预处理
│   └── explore_data.py     # 数据探索分析脚本
│
├──  models/               # 模型定义模块
│   ├── text_encoder.py     # 文本编码器（DistilBERT）
│   ├── image_encoder.py    # 图像编码器（ResNet50，2048维）
│   ├── fusion.py           # 基础融合策略（Late/Early/Cross-Attn/Gated）
│   ├── advanced_fusion.py  # 高级融合（Aligned/Hierarchical）
│   ├── multimodal_model.py # 完整多模态分类器
│   └── train_utils.py      # 模型训练辅助函数
│
├──  utils/                # 工具函数模块
│   ├── train_utils.py      # 训练工具（EarlyStopping、metrics、checkpoint）
│   ├── experiment_logger.py # 实验日志（CSV格式记录结果）
│   ├── logger.py           # 通用日志工具
│   └── visualize.py        # 可视化辅助函数
│
├──  scripts/              # 实验运行脚本
│   ├── run_experiment_simple.py        # Phase 1-3: 消融+融合对比
│   ├── run_experiment_optimized.py     # Phase 4: 解冻编码器优化
│   ├── run_data_preprocessing_exp.py   # Phase 5: 数据预处理实验
│   ├── run_practical_optimization.py   # Phase 6: 超参数搜索
│   ├── run_hp1_improvements.py         # Phase 7: Bad Case改进
│   ├── run_correct_analysis.py         # Bad Case分析+测试集预测
│   └── run_experiments.py              # 统一实验入口
│
├──  visualization/        # 可视化脚本
│   ├── visualize_experiments.py        # 实验结果可视化（所有图表）
│   ├── generate_report_figures.py      # 生成报告专用图表
│   ├── generate_figures_v3.py          # 高级图表生成
│   └── generate_overfit_fig.py         # 过拟合分析图
├──  experiments/          # 实验结果（自动生成）
│   ├── all_results.csv     # 所有实验结果汇总
│   ├── checkpoints/        # 模型权重文件
│   │   └── HP1_BEST_best.pth  # 最佳模型（72.25%）
│   ├── hyperparam_checkpoint.json      # 超参数搜索进度
│   ├── data_aug_checkpoint.json        # 数据预处理实验进度
│   └── badcase_improvement_checkpoint.json  # Bad Case改进进度
│
├──  analysis_results/     # 分析结果
│   ├── bad_cases_hp1_best.csv         # HP1模型错误样本
│   └── bad_cases_detailed.csv         # 详细错误分析
│
├──  figures/              # 实验图表（自动生成）
│   ├── fig1_ablation_study.png        # 消融实验对比
│   ├── fig2_fusion_comparison.png     # 融合方法对比
│   ├── fig4_confusion_matrix.png      # 混淆矩阵
│   └── ...
│
├──  splits/               # 数据划分（自动生成）
│   ├── train_split.txt     # 训练集GUID列表
│   ├── val_split.txt       # 验证集GUID列表
│   └── split_info.json     # 划分信息
│
├── train.py                 # 核心训练脚本
├── predict.py               # 预测脚本
├── evaluate.py              # 评估脚本
├── predictions.txt          # 测试集预测结果
├── README.md                # 项目说明文档
└── requirements.txt         # Python依赖列表
```

##  代码使用教程

### 方式1: 复现最佳结果

```bash
# 训练最佳模型
python scripts/run_practical_optimization.py --hp1-only

# 使用训练好的模型预测测试集
python scripts/run_correct_analysis.py --predict --model "experiments/checkpoints/HP1_BEST_best.pth"

# 查看预测结果
cat predictions.txt  # Linux/macOS
type predictions.txt  # Windows
```

### 方式2: 使用已保存模型直接预测

```bash
# 如果已有训练好的模型，直接预测
python scripts/run_correct_analysis.py --predict --model "experiments/checkpoints/HP1_BEST_best.pth"
```

### 方式3: 运行完整实验流程

```bash
# Phase 1-3: 基础实验（消融+融合对比）
python scripts/run_experiment_simple.py --all

# Phase 4: 优化训练（解冻编码器）
python scripts/run_experiment_optimized.py --run

# Phase 5: 数据预处理实验
python scripts/run_data_preprocessing_exp.py --run

# Phase 6: 超参数搜索
python scripts/run_practical_optimization.py --hyperparam

# Phase 7: Bad Case分析
python scripts/run_correct_analysis.py --badcase --model "experiments/checkpoints/HP1_BEST_best.pth"

# 生成可视化图表
python visualization/visualize_experiments.py
```

---

##  完整实验流程

### Phase 1: 消融实验（验证多模态有效性）

```bash
python run_experiment_simple.py --phase 1
```

实验内容：
- A1: Multimodal (Late Fusion)
- A2: Text-Only
- A3: Image-Only

### Phase 2: 融合方法对比

```bash
python run_experiment_simple.py --phase 2
```

实验内容：
- F1: Late Fusion
- F2: Early Fusion
- F3: Cross-Attention Fusion
- F4: Gated Fusion

### Phase 3: 高级融合方法

```bash
python run_experiment_simple.py --phase 3
```

实验内容：
- AF1: Aligned Fusion（特征对齐）
- AF2: Hierarchical Fusion（分层融合）

### Phase 4: 优化训练（解冻编码器）

```bash
python run_experiment_optimized.py --run
```

关键优化：
- 解冻 DistilBERT 最后2层
- 解冻 ResNet50 layer4
- 分层学习率：预训练层1e-5，融合层5e-5，分类层1e-4

### Phase 5: 数据预处理实验

```bash
python run_data_preprocessing_exp.py --run
```

实验内容：
- DA1: 基线（无预处理）
- DA2: 仅文本清洗（移除URL、@mentions等）
- DA3: 仅图像增强（RandomCrop、ColorJitter、Flip）
- DA4: 文本清洗 + 图像增强

### Phase 6: 超参数搜索

```bash
python run_practical_optimization.py --hyperparam
```

搜索空间：
- Dropout: [0.2, 0.25, 0.3]
- LR_Classifier: [5e-5, 1e-4, 2e-4]
- Weight_Decay: [0.01, 0.015, 0.02]

### Phase 7: Bad Case分析与改进

```bash
# 1. Bad Case分析
python run_correct_analysis.py --badcase --model "experiments/checkpoints/HP1_BEST_best.pth"

# 2. 针对性改进实验（类别权重调整、Focal Loss等）
python run_hp1_improvements.py --run
```

### Phase 8: 生成最终预测

```bash
python run_correct_analysis.py --predict --model "experiments/checkpoints/HP1_BEST_best.pth"
```

---

##  实验结果

### 1. 消融实验（Phase 1）

| 实验ID | 模态 | Val Acc | Val F1 | 结论 |
|--------|------|---------|--------|------|
| A1 | Multimodal | 67.00% | 0.565 | 基线 |
| A2 | Text-Only | 64.75% | 0.533 | -2.25% |
| A3 | Image-Only | 62.62% | 0.344 | -4.38% |

**结论**：多模态融合比单模态提升2-4%，验证了融合的有效性。

### 2. 融合方法对比（Phase 2-3）

| 实验ID | 融合方法 | Val Acc | Val F1 |
|--------|----------|---------|--------|
| F1 | Late Fusion | 67.00% | 0.565 |
| F3 | Cross-Attention | 66.75% | 0.566 |
| AF1 | Aligned Fusion | 66.75% | 0.567 |
| AF2 | Hierarchical | 64.75% | 0.516 |
| F2 | Early Fusion | 64.12% | 0.373 |
| F4 | Gated Fusion | 61.00% | 0.281 |

**结论**：简单融合（Late、Cross-Attention）效果最好，复杂融合容易过拟合。

### 3. 优化训练（Phase 4）

| 实验ID | 配置 | Val Acc | 提升 |
|--------|------|---------|------|
| OPT_late | Late + 解冻 | 71.13% | +4.13% |
| OPT_cross_attn | Cross-Attn + 解冻 | 71.25% | +4.25% |

**结论**：解冻编码器最后几层 + 分层学习率是有效的优化策略。

### 4. 数据预处理实验（Phase 5）

| 实验ID | 预处理方式 | Val Acc | Val F1 |
|--------|-----------|---------|--------|
| DA1 | 基线（无预处理） | **71.37%** | 0.602 |
| DA3 | 仅图像增强 | 71.13% | 0.605 |
| DA4 | 文本+图像 | 70.63% | 0.574 |
| DA2 | 仅文本清洗 | 70.00% | 0.594 |

**结论**：数据预处理对本数据集效果不明显，预训练模型已具有足够的鲁棒性。

### 5. 超参数搜索（Phase 6）

| 配置 | Dropout | LR_Clf | Weight_Decay | Val Acc |
|------|---------|--------|--------------|---------|
| **HP1** | **0.2** | **1e-4** | **0.01** | **72.25%** |
| HP4 | 0.3 | 2e-4 | 0.01 | 71.75% |
| HP5 | 0.25 | 1e-4 | 0.015 | 71.63% |
| HP2 | 0.3 | 1e-4 | 0.02 | 71.50% |
| HP3 | 0.3 | 5e-5 | 0.01 | 70.63% |

**最佳配置**：Dropout=0.2, LR_Classifier=1e-4, Weight_Decay=0.01

### 6. Bad Case分析（Phase 7）

**错误分布（共222个错误/800验证样本）**：

| 错误类型 | 数量 | 占比 |
|----------|------|------|
| positive → neutral | 65 | 29.3% |
| neutral → positive | 61 | 27.5% |
| negative → positive | 45 | 20.3% |
| negative → neutral | 18 | 8.1% |
| positive → negative | 18 | 8.1% |
| neutral → negative | 15 | 6.8% |

**主要问题**：positive 和 neutral 互相混淆（56.8%）

**改进尝试**：
- 类别权重调整  效果不明显
- Focal Loss  效果不明显

**分析**：小数据集上继续优化的边际收益低

### 7. 最终结果汇总

| 阶段 | 最佳配置 | Val Acc |
|------|----------|---------|
| 基线（冻结编码器） | Late Fusion | 67.00% |
| 解冻编码器 | Cross-Attention | 71.25% |
| **超参数优化** | **HP1** | **72.25%** |

**总提升**：67.00% → 72.25%（+5.25%）

---

##  模型架构

### Cross-Attention Fusion + 解冻编码器

```
┌─────────────────────────────────────────────────────────────────┐
│                    输入层                                        │
│  Text: "I love this product!"    Image: [224x224x3]             │
└──────────────────┬──────────────────────┬───────────────────────┘
                   ↓                      ↓
┌──────────────────────────────┐ ┌────────────────────────────────┐
│     DistilBERT Encoder       │ │      ResNet50 Encoder          │
│  (解冻最后2层transformer)     │ │    (解冻layer4)                │
│     [CLS] → 768维             │ │    Global Avg Pool → 2048维   │
└──────────────────┬───────────┘ └────────────────┬───────────────┘
                   ↓                              ↓
              Linear(768→512)               Linear(2048→512)
                   ↓                              ↓
              Text Features [512]           Image Features [512]
                   ↓                              ↓
┌──────────────────────────────────────────────────────────────────┐
│                 Cross-Attention Fusion                           │
│  Q = Text, K = Image, V = Image → Attended Text [512]           │
│  Q = Image, K = Text, V = Text → Attended Image [512]           │
│  Concat → [1024] → FC → [512]                                   │
└──────────────────────────────────────────────────────────────────┘
                   ↓
              Dropout(0.2)
                   ↓
            Linear(512→3)
                   ↓
         [positive, neutral, negative]
```

### 分层学习率设置

| 层 | 学习率 | 说明 |
|----|--------|------|
| 预训练编码器 | 1e-5 | 轻微微调 |
| 融合层 | 5e-5 | 中等学习 |
| 分类层 | 1e-4 | 快速学习 |

### 训练配置

```python
{
    'batch_size': 8,
    'accumulation_steps': 4,      # 等效batch_size=32
    'num_epochs': 20,
    'early_stopping_patience': 7,
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'class_weights': [1.0, 1.5, 3.0],  # positive, negative, neutral
}
```

##  参考资料

### 论文

1. **BERT**: Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (NAACL 2019)
2. **DistilBERT**: Sanh et al. "DistilBERT, a distilled version of BERT" (NeurIPS 2019 Workshop)
3. **ResNet**: He et al. "Deep Residual Learning for Image Recognition" (CVPR 2016)
4. **Attention**: Vaswani et al. "Attention Is All You Need" (NeurIPS 2017)
5. **Multimodal Fusion**: Baltrušaitis et al. "Multimodal Machine Learning: A Survey and Taxonomy" (TPAMI 2019)
6. **Focal Loss**: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)

### 代码参考

| 来源 | 用途 |
|------|------|
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | DistilBERT实现 |
| [PyTorch Vision](https://github.com/pytorch/vision) | ResNet50实现 |
| [GloGNN](https://github.com/RecklessRonan/GloGNN) | README格式参考 |

### 关键技术

- **迁移学习**：使用预训练的DistilBERT和ResNet50
- **分层学习率**：不同层使用不同学习率
- **梯度累积**：小batch模拟大batch训练
- **Early Stopping**：防止过拟合
- **类别加权**：处理数据不平衡

---

