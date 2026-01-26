"""
Training configuration
"""

class Config:
    """Base training configuration"""
    
    # Data paths
    DATA_DIR = r'D:\当代人工智能\project5\data'
    TRAIN_LABEL = r'D:\当代人工智能\project5\train.txt'
    TEST_LABEL = r'D:\当代人工智能\project5\test_without_label.txt'
    
    # Model
    NUM_CLASSES = 3
    MODEL_TYPE = 'multimodal'  # 'multimodal', 'text_only', 'image_only'
    MODALITY = 'multimodal'  # 'multimodal', 'text', 'image'
    TEXT_MODEL = 'distilbert-base-uncased'
    IMAGE_MODEL = 'resnet50'
    FUSION_TYPE = 'cross_attention'  # 'late', 'early', 'cross_attention'
    FEATURE_DIM = 512  # Standard dimension
    DROPOUT = 0.3
    FREEZE_ENCODERS = True
    
    # Training
    BATCH_SIZE = 4  # Reasonable batch size
    ACCUMULATION_STEPS = 8  # Effective batch = 4 × 8 = 32
    NUM_EPOCHS = 50  # Increased for better convergence
    LEARNING_RATE = 3e-5  # Moderate learning rate
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1  # 10% of total steps
    MAX_GRAD_NORM = 1.0
    
    # Data
    VAL_RATIO = 0.2  # 80/20划分，验证集800样本
    NUM_WORKERS = 0  # Windows: must be 0
    IMG_SIZE = 224  # Standard ResNet input size
    
    # Optimization
    USE_AMP = False  # Mixed precision (set True if GPU)
    GRADIENT_CHECKPOINTING = False  # Disabled to avoid warnings
    
    # Saving
    SAVE_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    EXPERIMENT_NAME = None  # Auto-generate: {fusion_type}_{modality}_YYYYMMDD_HHMMSS
    RESUME_FROM = None  # 从头开始训练 Cross-Attention
    SAVE_EVERY = 1  # Save every N epochs
    EARLY_STOPPING_PATIENCE = 5
    
    # Logging
    LOG_EVERY = 50  # Log every N steps
    EVAL_EVERY = 1  # Evaluate every N epochs
    
    # Device
    DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    
    # Random seed
    SEED = 42


def get_config():
    """Get default config with forced reload to avoid module caching"""
    # Force reload the module to pick up any changes made by batch scripts
    import importlib
    import sys
    if 'configs.config' in sys.modules:
        importlib.reload(sys.modules['configs.config'])
    return Config()
