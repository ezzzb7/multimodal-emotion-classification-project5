"""
数据加载器
负责创建训练/验证集划分和DataLoader
"""
import os
import json
import torch
from torch.utils.data import DataLoader
from .dataset import MultimodalDataset
from .preprocessing import TextPreprocessor, ImagePreprocessor

def create_data_splits(train_label_file, val_ratio=0.15, seed=42, save_dir='splits'):
    """
    创建训练集/验证集划分并保存到本地
    
    Args:
        train_label_file: 原始训练标签文件路径
        val_ratio: 验证集比例
        seed: 随机种子
        save_dir: 保存划分结果的目录
    
    Returns:
        train_df, val_df: 划分后的DataFrame
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    print(f"\n{'='*60}")
    print(f"创建数据集划分 (验证集比例: {val_ratio}, 随机种子: {seed})")
    print(f"{'='*60}")
    
    # 读取原始标签文件
    df = pd.read_csv(train_label_file, sep=',', header=None, names=['guid', 'tag'])
    
    # 过滤掉有问题的行（如果有'tag'作为标签值）
    df = df[df['tag'].isin(['positive', 'negative', 'neutral'])]
    
    print(f"总样本数: {len(df)}")
    print(f"标签分布:\n{df['tag'].value_counts()}\n")
    
    # 分层划分（保持各类别比例）
    train_df, val_df = train_test_split(
        df, 
        test_size=val_ratio, 
        random_state=seed,
        stratify=df['tag']  # 按标签分层
    )
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存划分结果
    train_split_path = os.path.join(save_dir, 'train_split.txt')
    val_split_path = os.path.join(save_dir, 'val_split.txt')
    
    train_df.to_csv(train_split_path, index=False, header=False)
    val_df.to_csv(val_split_path, index=False, header=False)
    
    # 保存划分信息
    split_info = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'val_ratio': val_ratio,
        'seed': seed,
        'train_label_distribution': train_df['tag'].value_counts().to_dict(),
        'val_label_distribution': val_df['tag'].value_counts().to_dict()
    }
    
    info_path = os.path.join(save_dir, 'split_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    print(f"训练集: {len(train_df)} 样本")
    print(f"  标签分布: {train_df['tag'].value_counts().to_dict()}")
    print(f"\n验证集: {len(val_df)} 样本")
    print(f"  标签分布: {val_df['tag'].value_counts().to_dict()}")
    print(f"\n划分结果已保存到: {save_dir}/")
    print(f"{'='*60}\n")
    
    return train_df, val_df


def get_data_loaders(data_dir, 
                     train_label_file,
                     test_label_file=None,
                     batch_size=8,
                     val_ratio=0.15,
                     num_workers=0,
                     seed=42,
                     img_size=224,
                     force_resplit=False):
    """
    获取训练/验证/测试的DataLoader
    
    Args:
        data_dir: 数据目录（包含所有.txt和.jpg文件）
        train_label_file: 训练标签文件路径
        test_label_file: 测试标签文件路径（可选）
        batch_size: 批次大小（默认16，内存不足可降至4或8）
        val_ratio: 验证集比例
        num_workers: 数据加载线程数（Windows必须为0）
        seed: 随机种子
        img_size: 图像尺寸
        force_resplit: 是否强制重新划分数据集
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # === 创建或加载数据划分 ===
    split_dir = 'splits'
    train_split_path = os.path.join(split_dir, 'train_split.txt')
    val_split_path = os.path.join(split_dir, 'val_split.txt')
    
    if force_resplit or not os.path.exists(train_split_path):
        create_data_splits(train_label_file, val_ratio, seed, split_dir)
    else:
        print(f"使用已有的数据划分: {split_dir}/")
    
    # === 创建预处理器 ===
    # 训练集：使用数据增强
    train_text_prep = TextPreprocessor(
        remove_emoji=False, 
        lowercase=True, 
        remove_hashtags=False
    )
    train_img_prep = ImagePreprocessor(mode='train', img_size=img_size)
    
    # 验证/测试集：不使用数据增强
    val_text_prep = TextPreprocessor(
        remove_emoji=False, 
        lowercase=True, 
        remove_hashtags=False
    )
    val_img_prep = ImagePreprocessor(mode='val', img_size=img_size)
    
    # === 创建数据集 ===
    print(f"\n{'='*60}")
    print(f"创建数据集 (batch_size={batch_size}, num_workers={num_workers})")
    print(f"{'='*60}")
    
    train_dataset = MultimodalDataset(
        data_dir=data_dir,
        label_file=train_split_path,
        text_preprocessor=train_text_prep,
        image_preprocessor=train_img_prep,
        mode='train'
    )
    
    val_dataset = MultimodalDataset(
        data_dir=data_dir,
        label_file=val_split_path,
        text_preprocessor=val_text_prep,
        image_preprocessor=val_img_prep,
        mode='val'
    )
    
    # === 创建数据加载器 ===
    # Windows优化：强制单线程、禁用pin_memory以最小化内存占用
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 强制单线程
        pin_memory=False,  # 禁用以节省内存
        drop_last=True,
        persistent_workers=False  # 禁用持久化worker
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 强制单线程
        pin_memory=False,  # 禁用以节省内存
        persistent_workers=False
    )
    
    print(f"训练 DataLoader: {len(train_loader)} batches")
    print(f"验证 DataLoader: {len(val_loader)} batches")
    
    # === 测试集（如果提供）===
    test_loader = None
    if test_label_file and os.path.exists(test_label_file):
        test_dataset = MultimodalDataset(
            data_dir=data_dir,
            label_file=test_label_file,
            text_preprocessor=val_text_prep,
            image_preprocessor=val_img_prep,
            mode='test'
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False
        )
        print(f"测试 DataLoader: {len(test_loader)} batches")
    
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, test_loader


def collate_fn(batch):
    """
    自定义collate函数，用于处理变长文本
    如果后续使用BERT等模型，可以在这里进行tokenization
    """
    guids = [item['guid'] for item in batch]
    texts = [item['text'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    return {
        'guid': guids,
        'text': texts,
        'image': images,
        'label': labels
    }
