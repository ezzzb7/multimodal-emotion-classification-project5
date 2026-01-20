# 多模态情感分类 (Multimodal Sentiment Classification)

[![GitHub](https://img.shields.io/badge/GitHub-multimodal--emotion--classification-blue)](https://github.com/ezzzb7/multimodal-emotion-classification-project5)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)

基于文本和图像的多模态情感分类系统，支持三种融合策略（Late Fusion、Early Fusion、Cross-Attention）和消融实验。

## 📋 目录

- [项目简介](#项目简介)
- [环境配置](#环境配置)
- [代码结构](#代码结构)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [完整实验流程](#完整实验流程)
- [实验结果](#实验结果)
- [模型架构](#模型架构)
- [参考资料](#参考资料)

## 🎯 项目简介

本项目实现了一个多模态情感分类系统，给定配对的文本和图像，预测对应的情感标签（positive、neutral、negative）。

### 主要特性

- **多种融合策略**：Late Fusion、Early Fusion、Cross-Attention Fusion
- **消融实验**：Text-Only、Image-Only模型对比
- **断点续传**：支持训练中断后继续训练
- **完整日志**：详细的训练日志和可视化
- **内存优化**：针对资源受限环境优化
- **Git管理**：完整的版本控制和实验追踪

### 性能指标

| 模型 | 验证集准确率 | F1-Score |
|------|------------|----------|
| **Late Fusion (Baseline)** | 67.5% | 0.5856 |
| Text-Only | TBD | TBD |
| Image-Only | TBD | TBD |
| Early Fusion | TBD | TBD |
| Cross-Attention | TBD | TBD |

> **注**：目前基线模型已训练10轮，建议继续训练至收敛（预计70-72%）。

## 🔧 环境配置

### 系统要求

- Python 3.8+
- Windows/Linux/macOS
- CPU/GPU（建议4GB+ RAM）

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/ezzzb7/multimodal-emotion-classification-project5.git
cd multimodal-emotion-classification-project5

# 安装依赖
pip install -r requirements.txt
```

### requirements.txt

```txt
torch>=1.12.0
torchvision>=0.13.0
transformers>=4.20.0
pillow>=9.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tqdm>=4.62.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## 📁 代码结构

```
code/
├── configs/
│   └── config.py              # 训练配置（超参数、路径等）
├── data/
│   ├── __init__.py
│   ├── preprocessing.py       # 数据预处理
│   └── data_loader.py         # 数据加载器
├── models/
│   ├── __init__.py
│   ├── text_encoder.py        # 文本编码器（DistilBERT）
│   ├── image_encoder.py       # 图像编码器（ResNet50）
│   ├── fusion.py              # 融合策略（Late/Early/Cross-Attention）
│   └── multimodal_model.py    # 完整模型（含消融实验模型）
├── utils/
│   ├── logger.py              # 训练日志记录
│   ├── train_utils.py         # 训练工具函数
│   └── visualize.py           # 可视化工具
├── checkpoints/               # 模型checkpoint
├── logs/                      # 训练日志
├── splits/                    # 数据划分
├── train.py                   # 训练主脚本
├── train_all_experiments.py   # 自动化实验脚本
├── resume_training.py         # 断点续传工具
├── predict.py                 # 测试集预测
├── evaluate.py                # 模型评估
├── run_experiments.py         # 实验管理菜单
├── CHECKPOINT_NAMING.md       # Checkpoint命名规范
└── README.md                  # 本文件
```

## 📊 数据准备

### 数据目录结构

```
D:\当代人工智能\project5\
├── data/                      # 所有图像和文本文件
│   ├── 10001.txt
│   ├── 10001.jpg
│   ├── 10002.txt
│   ├── 10002.jpg
│   └── ...
├── train.txt                  # 训练标签（guid,tag）
└── test_without_label.txt     # 测试标签（guid,null）
```

### 数据格式

**train.txt**:
```
guid,tag
10001,positive
10002,negative
10003,neutral
...
```

**test_without_label.txt**:
```
guid,tag
20001,null
20002,null
...
```

## 🚀 快速开始

### 1. 继续训练基线模型（已训练10轮）

```bash
# 使用交互式工具选择checkpoint
python resume_training.py

# 或直接指定checkpoint
python train.py
# 在config.py中设置: RESUME_FROM = 'checkpoints/late_multimodal_20260118_180829_epoch10.pth'
```

### 2. 运行单个实验

```bash
# 训练Text-Only模型
python train_all_experiments.py --single text

# 训练Image-Only模型
python train_all_experiments.py --single image

# 训练Early Fusion模型
python train_all_experiments.py --single early

# 训练Cross-Attention模型
python train_all_experiments.py --single cross_attention
```

### 3. 预测测试集

```bash
# 使用最佳模型预测
python predict.py --checkpoint checkpoints/best_late_multimodal_20260118_180829.pth --output predictions.txt
```

### 4. 生成可视化

```bash
# 为指定实验生成图表
python utils/visualize.py logs/late_multimodal_20260118_180829
```

## 🔬 完整实验流程

### 方案一：自动化运行所有实验

```bash
# 完整训练模式（跳过已完成的基线）
python train_all_experiments.py --skip-baseline

# 快速测试模式（每个实验只训练10轮）
python train_all_experiments.py --quick
```

实验顺序：
1. ~~基线模型 (Late Fusion)~~ ✓ 已完成10轮
2. Text-Only消融实验
3. Image-Only消融实验
4. Early Fusion高级融合
5. Cross-Attention高级融合

### 方案二：手动运行实验

#### 实验1：继续基线训练（从第11轮开始）

```bash
# 配置：configs/config.py
MODEL_TYPE = 'multimodal'
FUSION_TYPE = 'late'
MODALITY = 'multimodal'
NUM_EPOCHS = 100  # 或20、30等
RESUME_FROM = 'checkpoints/late_multimodal_20260118_180829_epoch10.pth'

# 运行
python train.py
```

#### 实验2：Text-Only消融实验

```bash
# 配置：configs/config.py
MODEL_TYPE = 'text_only'
MODALITY = 'text'
RESUME_FROM = None

# 运行
python train.py
```

#### 实验3：Image-Only消融实验

```bash
# 配置：configs/config.py
MODEL_TYPE = 'image_only'
MODALITY = 'image'
RESUME_FROM = None

# 运行
python train.py
```

#### 实验4：Early Fusion

```bash
# 配置：configs/config.py
MODEL_TYPE = 'multimodal'
FUSION_TYPE = 'early'
MODALITY = 'multimodal'
RESUME_FROM = None

# 运行
python train.py
```

#### 实验5：Cross-Attention Fusion

```bash
# 配置：configs/config.py
MODEL_TYPE = 'multimodal'
FUSION_TYPE = 'cross_attention'
MODALITY = 'multimodal'
RESUME_FROM = None

# 运行
python train.py
```

### 预测和评估

```bash
# 1. 预测测试集（选择最佳模型）
python predict.py \
    --checkpoint checkpoints/best_late_multimodal_20260118_180829.pth \
    --output predictions_late_fusion.txt

# 2. 评估验证集（可选）
python evaluate.py --checkpoint checkpoints/best_late_multimodal_20260118_180829.pth

# 3. 生成可视化
python utils/visualize.py logs/late_multimodal_20260118_180829
python utils/visualize.py logs/text_only_20260118_190000
python utils/visualize.py logs/image_only_20260118_200000
# ... 为每个实验生成
```

## 📈 实验结果

### 当前进度

- [x] 数据处理和加载器
- [x] 基线Late Fusion模型（10/100轮，67.5% val acc）
- [ ] Text-Only消融实验
- [ ] Image-Only消融实验
- [ ] Early Fusion高级融合
- [ ] Cross-Attention高级融合
- [ ] 测试集预测
- [ ] 实验报告和可视化

### Checkpoint管理

所有checkpoint保存在`checkpoints/`目录，命名规范：

```
best_{model_type}_{timestamp}.pth          # 最佳模型
{model_type}_{timestamp}_epoch{N}.pth      # 周期性保存
```

示例：
- `best_late_multimodal_20260118_180829.pth` - 基线最佳模型（67.5%）
- `late_multimodal_20260118_180829_epoch10.pth` - 第10轮checkpoint
- `best_text_only_20260118_190000.pth` - Text-Only最佳模型

详见 [CHECKPOINT_NAMING.md](CHECKPOINT_NAMING.md)

### 日志和可视化

每个实验生成独立日志：

```
logs/{experiment_name}/
├── config.json           # 实验配置
├── training_log.json     # JSON格式日志
├── training_log.csv      # CSV格式日志
├── step_log.txt          # 详细步骤日志
├── error_samples.json    # 错误样本分析
└── plots/                # 可视化图表（运行visualize.py后生成）
    ├── loss_curve.png
    ├── accuracy_curve.png
    ├── f1_curve.png
    └── confusion_matrix.png
```

## 🧠 模型架构

### Late Fusion (Baseline)

```
Text Input → DistilBERT → [768] → FC → [512] ─┐
                                                ├→ Concat [1024] → Classifier → [3]
Image Input → ResNet50 → [2048] → FC → [512] ─┘
```

**特点**：
- 简单有效的基线方法
- 独立提取文本和图像特征后拼接
- 参数量：91M frozen + 1.7M trainable (1.86%)

### Early Fusion

```
Text Input → DistilBERT → [768] → Project → [512] ─┐
                                                     ├→ Element-wise + → Fusion → [512] → Classifier → [3]
Image Input → ResNet50 → [2048] → Project → [512] ─┘
```

**特点**：
- 特征级融合，更紧密的多模态交互
- 使用element-wise操作（加法/乘法）
- 更少的融合后维度

### Cross-Attention Fusion

```
Text Features [512] ─────┐
                         ├→ Cross-Attention ─┐
Image Features [512] ────┘                   ├→ Fused [256] → Classifier → [3]
                                             │
Image Features [512] ─────┐                  │
                          ├→ Cross-Attention ─┘
Text Features [512] ──────┘
```

**特点**：
- 双向注意力机制：文本→图像，图像→文本
- 捕捉跨模态语义关联
- 最先进的融合方法

### Text-Only / Image-Only (Ablation)

```
Text Only:  Text Input → DistilBERT → [512] → Classifier → [3]
Image Only: Image Input → ResNet50 → [512] → Classifier → [3]
```

## 🐛 Bug记录与解决方案

### Bug 1: 固定59.67%准确率（模型总是预测positive）

**问题**：训练初期模型陷入局部最优，总是预测占比最大的类别。

**原因**：数据不平衡（59.7% positive, 29.8% negative, 10.5% neutral）

**解决**：
```python
# 使用加权交叉熵损失
class_weights = torch.tensor([0.34, 1.97, 0.69])  # 反比例权重
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**效果**：准确率从59.67%固定值提升至67.5%并持续改善。

### Bug 2: 内存不足（OOM）

**问题**：在资源受限环境下训练崩溃。

**解决**：
1. 冻结预训练编码器（91M参数）
2. 梯度累积（batch_size=4, accumulation=8）
3. 及时清理中间张量（`del`, `torch.cuda.empty_cache()`）
4. 禁用梯度检查点（避免兼容性警告）

### Bug 3: Checkpoint覆盖问题

**问题**：不同实验的checkpoint互相覆盖。

**解决**：实验特定命名 + 时间戳
```python
experiment_name = f"{fusion_type}_{modality}_{timestamp}"
```

## 📚 参考资料

### 论文

1. **Multimodal Sentiment Analysis**:
   - Zadeh et al. "Multimodal Sentiment Intensity Analysis in Videos" (2016)
   - Poria et al. "A Review of Affective Computing" (2017)

2. **Fusion Strategies**:
   - Baltrušaitis et al. "Multimodal Machine Learning: A Survey" (2019)
   - Late Fusion, Early Fusion, Hybrid Fusion比较

3. **Attention Mechanisms**:
   - Vaswani et al. "Attention Is All You Need" (2017)
   - Cross-modal Attention for multimodal learning

### 代码参考

- Hugging Face Transformers: https://github.com/huggingface/transformers
- PyTorch Vision Models: https://github.com/pytorch/vision
- GloGNN README: https://github.com/RecklessRonan/GloGNN

### 模型

- **Text Encoder**: DistilBERT ([distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased))
- **Image Encoder**: ResNet50 (ImageNet pretrained)

## 🤝 贡献

欢迎提Issue和PR！

## 📄 许可

MIT License

## 👤 作者

- GitHub: [@ezzzb7](https://github.com/ezzzb7)
- 项目: [multimodal-emotion-classification-project5](https://github.com/ezzzb7/multimodal-emotion-classification-project5)

---

**更新日志**

- 2026-01-18: 
  - ✅ 基线Late Fusion训练10轮（67.5% val acc）
  - ✅ 实现所有融合策略和消融实验模型
  - ✅ 完整的checkpoint管理和断点续传
  - ✅ 自动化实验训练脚本
  - 📝 待完成：消融实验、高级融合、测试集预测
