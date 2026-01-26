"""
统一实验配置
所有实验共享的超参数，确保公平对比
"""
import os
from datetime import datetime


class ExperimentConfig:
    """实验基础配置 - 所有实验共享"""
    
    # ========== 数据路径 ==========
    DATA_DIR = r'D:\当代人工智能\project5\data'
    TRAIN_LABEL = r'D:\当代人工智能\project5\train.txt'
    TEST_LABEL = r'D:\当代人工智能\project5\test_without_label.txt'
    
    # ========== 固定超参数（公平对比） ==========
    SEED = 42                    # 随机种子，确保可复现
    VAL_RATIO = 0.2              # 验证集比例 (80/20划分)
    NUM_CLASSES = 3              # 三分类
    
    # 训练参数
    BATCH_SIZE = 8               # 批次大小
    ACCUMULATION_STEPS = 4       # 梯度累积 (effective batch = 32)
    NUM_EPOCHS = 30              # 最大训练轮次
    LEARNING_RATE = 2e-5         # 学习率
    WEIGHT_DECAY = 0.01          # 权重衰减
    WARMUP_RATIO = 0.1           # warmup比例
    MAX_GRAD_NORM = 1.0          # 梯度裁剪
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5  # 耐心值
    
    # 模型参数
    FEATURE_DIM = 512            # 特征维度
    DROPOUT = 0.3                # Dropout率
    FREEZE_ENCODERS = True       # 冻结编码器（小数据集必须）
    
    # 图像参数
    IMG_SIZE = 224               # 图像尺寸
    
    # 系统参数
    NUM_WORKERS = 0              # Windows必须为0
    DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    
    # 保存路径
    SAVE_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    EXPERIMENT_DIR = 'experiments'
    
    # 日志
    LOG_EVERY = 50
    EVAL_EVERY = 1
    SAVE_EVERY = 5
    
    @classmethod
    def get_experiment_config(cls, 
                              exp_id: str,
                              exp_name: str,
                              fusion_type: str = 'late',
                              modality: str = 'multimodal',
                              text_model: str = 'distilbert-base-uncased',
                              image_model: str = 'resnet50',
                              use_augmentation: bool = False,
                              **kwargs):
        """
        获取特定实验的配置
        
        Args:
            exp_id: 实验ID (如 'E1.1')
            exp_name: 实验名称 (如 'baseline_late_fusion')
            fusion_type: 融合类型 ('late', 'early', 'cross_attention', 'gated', 'clip')
            modality: 模态 ('multimodal', 'text', 'image')
            text_model: 文本模型
            image_model: 图像模型
            use_augmentation: 是否使用数据增强
            **kwargs: 其他覆盖参数
        """
        config = {
            # 实验标识
            'exp_id': exp_id,
            'exp_name': exp_name,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            
            # 实验变量
            'fusion_type': fusion_type,
            'modality': modality,
            'text_model': text_model,
            'image_model': image_model,
            'use_augmentation': use_augmentation,
            
            # 固定参数（从类属性）
            'seed': cls.SEED,
            'val_ratio': cls.VAL_RATIO,
            'num_classes': cls.NUM_CLASSES,
            'batch_size': cls.BATCH_SIZE,
            'accumulation_steps': cls.ACCUMULATION_STEPS,
            'num_epochs': cls.NUM_EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'warmup_ratio': cls.WARMUP_RATIO,
            'max_grad_norm': cls.MAX_GRAD_NORM,
            'early_stopping_patience': cls.EARLY_STOPPING_PATIENCE,
            'feature_dim': cls.FEATURE_DIM,
            'dropout': cls.DROPOUT,
            'freeze_encoders': cls.FREEZE_ENCODERS,
            'img_size': cls.IMG_SIZE,
            
            # 路径
            'data_dir': cls.DATA_DIR,
            'train_label': cls.TRAIN_LABEL,
            'test_label': cls.TEST_LABEL,
            'save_dir': cls.SAVE_DIR,
            'log_dir': cls.LOG_DIR,
            'experiment_dir': cls.EXPERIMENT_DIR,
        }
        
        # 覆盖参数
        config.update(kwargs)
        
        return config


# ========== 预定义实验配置 ==========

EXPERIMENTS = {
    # 阶段1: 基线和消融实验
    'E1.1': {
        'exp_name': 'baseline_late_fusion',
        'fusion_type': 'late',
        'modality': 'multimodal',
        'description': '基线模型：Late Fusion多模态'
    },
    'E1.2': {
        'exp_name': 'ablation_text_only',
        'fusion_type': 'none',
        'modality': 'text',
        'description': '消融实验：仅文本'
    },
    'E1.3': {
        'exp_name': 'ablation_image_only',
        'fusion_type': 'none',
        'modality': 'image',
        'description': '消融实验：仅图像'
    },
    
    # 阶段2: 融合策略对比
    'E2.1': {
        'exp_name': 'fusion_late',
        'fusion_type': 'late',
        'modality': 'multimodal',
        'description': '融合对比：Late Fusion'
    },
    'E2.2': {
        'exp_name': 'fusion_early',
        'fusion_type': 'early',
        'modality': 'multimodal',
        'description': '融合对比：Early Fusion'
    },
    'E2.3': {
        'exp_name': 'fusion_cross_attention',
        'fusion_type': 'cross_attention',
        'modality': 'multimodal',
        'description': '融合对比：Cross-Attention'
    },
    'E2.4': {
        'exp_name': 'fusion_gated',
        'fusion_type': 'gated',
        'modality': 'multimodal',
        'description': '融合对比：Gated Fusion'
    },
    
    # 阶段3: 数据增强对比
    'E3.1': {
        'exp_name': 'aug_none',
        'fusion_type': 'early',
        'modality': 'multimodal',
        'use_augmentation': False,
        'description': '增强对比：无增强'
    },
    'E3.2': {
        'exp_name': 'aug_text',
        'fusion_type': 'early',
        'modality': 'multimodal',
        'use_augmentation': True,
        'augment_text': True,
        'augment_image': False,
        'description': '增强对比：文本增强'
    },
    'E3.3': {
        'exp_name': 'aug_image',
        'fusion_type': 'early',
        'modality': 'multimodal',
        'use_augmentation': True,
        'augment_text': False,
        'augment_image': True,
        'description': '增强对比：图像增强'
    },
    'E3.4': {
        'exp_name': 'aug_both',
        'fusion_type': 'early',
        'modality': 'multimodal',
        'use_augmentation': True,
        'augment_text': True,
        'augment_image': True,
        'description': '增强对比：双重增强'
    },
    
    # 阶段4: 模型改进
    'E4.1': {
        'exp_name': 'clip_fusion',
        'fusion_type': 'clip',
        'modality': 'multimodal',
        'text_model': 'clip',
        'image_model': 'clip',
        'description': '改进：CLIP特征融合'
    },
    'E4.2': {
        'exp_name': 'ensemble',
        'fusion_type': 'ensemble',
        'modality': 'multimodal',
        'description': '改进：模型集成'
    },
}


def get_experiment(exp_id: str):
    """获取预定义实验配置"""
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment ID: {exp_id}")
    
    exp_preset = EXPERIMENTS[exp_id]
    return ExperimentConfig.get_experiment_config(exp_id=exp_id, **exp_preset)


def list_experiments():
    """列出所有预定义实验"""
    print("\n" + "="*70)
    print("预定义实验列表")
    print("="*70)
    for exp_id, exp_info in EXPERIMENTS.items():
        print(f"  {exp_id}: {exp_info['exp_name']}")
        print(f"       {exp_info['description']}")
    print("="*70 + "\n")


if __name__ == '__main__':
    list_experiments()
