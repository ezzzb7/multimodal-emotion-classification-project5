# 多模态情感分类 (Multimodal Sentiment Classification)

[![GitHub](https://img.shields.io/badge/GitHub-multimodal--emotion--classification-blue)](https://github.com/ezzzb7/multimodal-emotion-classification-project5)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)

基于文本和图像的多模态情感分类系统，支持三种融合策略（Late Fusion、Early Fusion、Cross-Attention）和完整的消融实验。

## 📋 目录

- [项目简介](#项目简介)
- [环境配置](#环境配置)
- [代码结构](#代码结构)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [实验结果](#实验结果)
- [模型架构](#模型架构)
- [Bug记录与解决方案](#bug记录与解决方案)
- [参考资料](#参考资料)

## 🎯 项目简介

本项目实现了一个多模态情感分类系统，给定配对的文本和图像，预测对应的情感标签（positive、neutral、negative）。

### 主要特性

- **多种融合策略**：Late Fusion、Early Fusion、Cross-Attention Fusion
- **完整消融实验**：Text-Only、Image-Only模型对比
- **断点续传**：支持训练中断后继续训练
- **完整日志**：详细的训练日志和可视化
- **内存优化**：针对资源受限环境优化

### 性能指标

| 模型 | 验证集准确率 | 验证集F1 | 说明 |
|------|------------|---------|------|
| **Early Fusion** | **69.00%** | 0.6098 | ⭐ 最佳单模型 |
| Cross-Attention | 68.83% | 0.6155 | 稳定 |
| Late Fusion | 68.67% | 0.5706 | 基线 |
| Text-Only | 65.00% | 0.5422 | 消融实验 |
| Image-Only | 64.67% | 0.4337 | 消融实验 |
| **Ensemble** | **~70%** | - | 多模型集成预期 |

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
│   └── config.py                  # 训练配置（超参数、路径等）
├── data/
│   ├── __init__.py
│   ├── data_loader.py             # 数据加载器
│   ├── dataset.py                 # Dataset类定义
│   └── preprocessing.py           # 数据预处理
├── models/
│   ├── __init__.py
│   ├── text_encoder.py            # 文本编码器（DistilBERT）
│   ├── image_encoder.py           # 图像编码器（ResNet50）
│   ├── fusion.py                  # 融合策略（Late/Early/Cross-Attention）
│   └── multimodal_model.py        # 完整模型定义
├── utils/
│   ├── __init__.py
│   ├── logger.py                  # 训练日志记录
│   └── train_utils.py             # 训练工具函数
├── checkpoints/                   # 模型checkpoint (git忽略)
├── logs/                          # 训练日志 (git忽略)
├── splits/                        # 数据划分
├── train.py                       # 主训练脚本
├── predict.py                     # 测试集预测
├── evaluate.py                    # 模型评估
├── ensemble_predict.py            # 集成预测
├── analyze_bad_cases.py           # Bad Case分析
├── augment_bad_cases.py           # 数据增强
├── compare_fusion_methods.py      # 融合方法对比
├── visualize_training.py          # 可视化工具
├── README.md                      # 本文件
└── requirements.txt               # 依赖列表
```

## 📊 数据准备

### 数据目录结构

```
D:\当代人工智能\project5\
├── data/                          # 所有图像和文本文件
│   ├── 10001.txt
│   ├── 10001.jpg
│   ├── 10002.txt
│   ├── 10002.jpg
│   └── ...
├── train.txt                      # 训练标签（guid,tag）
└── test_without_label.txt         # 测试标签（guid,null）
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

### 1. 训练模型

```bash
# 修改 configs/config.py 选择融合策略
# FUSION_TYPE = 'late' / 'early' / 'cross_attention'
# MODEL_TYPE = 'multimodal' / 'text_only' / 'image_only'

# 运行训练
python train.py
```

### 2. 模型评估

```bash
python evaluate.py --checkpoint checkpoints/best_early_multimodal_20260120_195503.pth
```

### 3. 测试集预测

```bash
python predict.py --checkpoint checkpoints/best_early_multimodal_20260120_195503.pth --output predictions.txt
```

### 4. 集成预测 (推荐)

```bash
python ensemble_predict.py --output predictions_ensemble.txt
```

### 5. Bad Case分析 (只用于训练集，避免信息泄露)

```bash
python analyze_bad_cases.py --split train
```

## 📈 实验结果

### 主实验结果

| 实验 | 模型配置 | Val Acc | Val F1 | 可训练参数 |
|-----|---------|---------|--------|-----------|
| 1 | Late Fusion (冻结) | 68.67% | 0.5706 | 1.7M |
| 2 | Early Fusion (冻结) | **69.00%** | 0.6098 | 2.6M |
| 3 | Cross-Attention (冻结) | 68.83% | 0.6155 | 2.4M |

### 消融实验结果

| 模态 | Accuracy | F1 | 相比多模态 |
|-----|----------|-----|-----------|
| Multimodal | 69.00% | 0.6098 | 基准 |
| Text-Only | 65.00% | 0.5422 | -4.00% |
| Image-Only | 64.67% | 0.4337 | -4.33% |

**结论**：多模态融合比单模态提升约4%，验证了融合文本和图像信息的有效性。

### 关键发现

1. **冻结编码器是关键**：3400样本不足以微调90M+参数的预训练模型
2. **简单融合足够有效**：Early/Late/Cross-Attention准确率差距<0.5%
3. **文本贡献略高于图像**：Text-Only > Image-Only
4. **模型集成可进一步提升**：预期70-71%

## 🧠 模型架构

### Late Fusion (Baseline)

```
Text Input → DistilBERT → [768] → FC → [512] ─┐
                                               ├→ Concat [1024] → Classifier → [3]
Image Input → ResNet50 → [2048] → FC → [512] ─┘
```

### Early Fusion (Best Single Model)

```
Text Input → DistilBERT → [768] → Project → [512] ─┐
                                                    ├→ Add + Fusion → [512] → Classifier → [3]
Image Input → ResNet50 → [2048] → Project → [512] ─┘
```

### Cross-Attention Fusion

```
Text [512] ←─── Attention ←── Image [512]
      ↓                            ↓
   Attended Text            Attended Image
             └──── Concat ────┘
                    ↓
              Classifier → [3]
```

## 🐛 Bug记录与解决方案

### Bug 1: 模型总是预测positive（固定59.67%准确率）

**原因**：数据不平衡（positive占59.7%），模型陷入局部最优

**解决**：使用类别加权的交叉熵损失
```python
class_weights = 1.0 / class_counts
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Bug 2: 内存不足（OOM）

**解决**：
- 冻结预训练编码器
- 梯度累积（batch_size=4, accumulation=8）
- 及时清理中间张量

### Bug 3: Windows DataLoader多进程报错

**解决**：设置 `num_workers=0`

### Bug 4: 验证集Bad Case信息泄露

**问题**：使用验证集错误样本进行数据增强，导致验证准确率虚高

**解决**：
```bash
# 只分析训练集
python analyze_bad_cases.py --split train
```

### Bug 5: 解冻编码器导致严重过拟合

**现象**：训练准确率98%，验证准确率70%（差距28%）

**解决**：保持编码器冻结，只训练融合层和分类器

## 📚 参考资料

### 论文

1. **BERT**: Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
2. **ResNet**: He et al. "Deep Residual Learning for Image Recognition" (2016)
3. **Attention**: Vaswani et al. "Attention Is All You Need" (2017)
4. **Multimodal Fusion**: Baltrušaitis et al. "Multimodal Machine Learning: A Survey" (2019)

### 代码参考

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch Vision](https://github.com/pytorch/vision)
- [GloGNN](https://github.com/RecklessRonan/GloGNN) - README格式参考

## 📄 许可

MIT License

## 👤 作者

- GitHub: [@ezzzb7](https://github.com/ezzzb7)
- 项目地址: [multimodal-emotion-classification-project5](https://github.com/ezzzb7/multimodal-emotion-classification-project5)

---

**最后更新**: 2026-01-24
