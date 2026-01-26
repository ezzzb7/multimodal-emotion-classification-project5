# 实验五：多模态情感分类

基于文本和图像的多模态情感分类模型，融合 DistilBERT 和 ResNet50 预训练模型。

## 环境配置

```bash
# 创建虚拟环境
conda create -n multimodal python=3.9
conda activate multimodal

# 安装依赖
pip install -r requirements.txt
```

### 依赖版本

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- torchvision 0.15+

## 数据准备

```
data/
├── train.txt          # 训练标签 (guid,tag)
├── test_without_label.txt  # 测试集 (guid,tag)
└── data/              # 多媒体文件
    ├── 1.txt          # 文本
    ├── 1.jpg          # 图像
    └── ...
```

- **训练集**：4000 样本（80/20分层划分 → 训练3200 / 验证800）
- **测试集**：511 样本
- **类别分布**：positive 59.7%, negative 29.8%, neutral 10.5%

## 代码结构

```
code/
├── configs/                           # 配置文件
│   ├── config.py                      # 基础训练配置
│   └── experiment_config.py           # 实验配置
│
├── data/                              # 数据处理模块
│   ├── data_loader.py                 # 数据加载器
│   ├── dataset.py                     # Dataset类定义
│   └── preprocessing.py               # 文本/图像预处理
│
├── models/                            # 模型定义
│   ├── text_encoder.py                # 文本编码器（DistilBERT）
│   ├── image_encoder.py               # 图像编码器（ResNet50）
│   ├── fusion.py                      # 融合策略（Late/Early/Cross-Attn/Gated）
│   ├── advanced_fusion.py             # 高级融合（Aligned/Hierarchical）
│   └── multimodal_model.py            # 完整多模态模型
│
├── utils/                             # 工具函数
│   ├── train_utils.py                 # 训练工具（EarlyStopping等）
│   └── experiment_logger.py           # 实验日志记录器
│
├── experiments/                       # 实验结果（自动生成）
│   ├── all_results.csv                # 所有实验结果汇总
│   ├── checkpoints/                   # 模型权重
│   │   └── HP1_BEST_best.pth          # 最佳模型（72.25%）
│   └── *_checkpoint.json              # 断点续传文件
│
├── analysis_results/                  # Bad Case分析结果
├── splits/                            # 数据划分文件
├── predictions.txt                    # 测试集预测结果
│
├── # ========== 运行脚本 ==========
├── run_experiment_simple.py           # Phase 1-3: 消融+融合对比实验
├── run_experiment_optimized.py        # Phase 4: 优化训练（解冻编码器）
├── run_practical_optimization.py      # Phase 6: 超参数搜索
├── run_data_preprocessing_exp.py      # Phase 5: 数据预处理实验
├── run_hp1_improvements.py            # Phase 7: Bad Case改进实验
├── run_correct_analysis.py            # Bad Case分析 + 测试集预测
│
├── train.py                           # 基础训练脚本
├── predict.py                         # 预测脚本
├── evaluate.py                        # 评估脚本
├── visualize_experiments.py           # 实验结果可视化
│
├── README.md
└── requirements.txt
```

## 快速开始

### 一键复现最佳结果

```bash
# 1. 训练最佳模型（HP1配置）
python run_hp1_improvements.py --hp1-only

# 2. 生成测试集预测
python run_correct_analysis.py --predict --model "experiments/checkpoints/HP1_BEST_best.pth"
```

### 使用已有模型直接预测

```bash
python run_correct_analysis.py --predict --model "experiments/checkpoints/HP1_BEST_best.pth"
# 结果保存在 predictions.txt
```

## 完整实验流程

### Phase 1: 消融实验

```bash
python run_experiment_simple.py --phase 1
```

验证多模态融合的有效性（A1: Multimodal, A2: Text-Only, A3: Image-Only）

### Phase 2-3: 融合方法对比

```bash
python run_experiment_simple.py --phase 2
python run_experiment_simple.py --phase 3
```

对比不同融合策略（Late/Early/Cross-Attention/Gated/Aligned/Hierarchical）

### Phase 4: 优化训练

```bash
python run_experiment_optimized.py --run
```

解冻编码器最后几层 + 分层学习率

### Phase 5: 数据预处理实验

```bash
python run_data_preprocessing_exp.py --run
```

对比文本清洗、图像增强的效果

### Phase 6: 超参数搜索

```bash
python run_practical_optimization.py --hyperparam
```

搜索 Dropout、学习率、权重衰减

### Phase 7: Bad Case分析

```bash
python run_correct_analysis.py --badcase --model "experiments/checkpoints/HP1_BEST_best.pth"
```

### Phase 8: 生成最终预测

```bash
python run_correct_analysis.py --predict --model "experiments/checkpoints/HP1_BEST_best.pth"
```

## 实验结果

### 消融实验

| 实验 | 模态 | Val Acc |
|------|------|---------|
| A1 | Multimodal | 67.00% |
| A2 | Text-Only | 64.75% |
| A3 | Image-Only | 62.62% |

### 融合方法对比

| 融合方法 | Val Acc |
|----------|---------|
| Late Fusion | 67.00% |
| Cross-Attention | 66.75% |
| Aligned Fusion | 66.75% |
| Hierarchical | 64.75% |
| Early Fusion | 64.12% |
| Gated Fusion | 61.00% |

### 优化训练

| 配置 | Val Acc |
|------|---------|
| Late + 解冻编码器 | 71.13% |
| Cross-Attn + 解冻编码器 | 71.25% |

### 超参数搜索

| 配置 | Dropout | Val Acc |
|------|---------|---------|
| **HP1** | **0.2** | **72.25%** |
| HP4 | 0.3 | 71.75% |
| HP5 | 0.25 | 71.63% |

### 最终结果

| 阶段 | Val Acc |
|------|---------|
| 基线（冻结编码器） | 67.00% |
| 解冻编码器 | 71.25% |
| **超参数优化** | **72.25%** |

**总提升**：67.00% → 72.25%（+5.25%）

## 模型架构

```
Text → DistilBERT (解冻最后2层) → Linear → 512维
                                              ↓
                                    Cross-Attention Fusion → 512维 → Classifier → 3类
                                              ↑
Image → ResNet50 (解冻layer4) → Linear → 512维
```

### 训练配置

| 参数 | 值 |
|------|-----|
| Batch Size | 8 (累积4步=32) |
| Epochs | 20 |
| Early Stopping | 7 |
| LR (预训练层) | 1e-5 |
| LR (融合层) | 5e-5 |
| LR (分类层) | 1e-4 |
| Dropout | 0.2 |
| Class Weights | [1.0, 1.5, 3.0] |

## 参考资料

### 论文

1. Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (NAACL 2019)
2. Sanh et al. "DistilBERT, a distilled version of BERT" (NeurIPS 2019)
3. He et al. "Deep Residual Learning for Image Recognition" (CVPR 2016)
4. Vaswani et al. "Attention Is All You Need" (NeurIPS 2017)
5. Baltrušaitis et al. "Multimodal Machine Learning: A Survey and Taxonomy" (TPAMI 2019)

### 代码参考

- [Hugging Face Transformers](https://github.com/huggingface/transformers) - DistilBERT
- [PyTorch Vision](https://github.com/pytorch/vision) - ResNet50

