"""
Predict sentiment labels for test set
预测测试集的情感标签
"""

import sys
import os
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from configs.config import get_config
from data.data_loader import get_data_loaders
from models.multimodal_model import MultimodalClassifier, TextOnlyClassifier, ImageOnlyClassifier


def predict_test_set(checkpoint_path, output_file='predictions.txt', model_type='multimodal'):
    """
    预测测试集并保存结果
    
    Args:
        checkpoint_path: 模型checkpoint路径
        output_file: 输出文件路径
        model_type: 'multimodal', 'text_only', 'image_only'
    """
    config = get_config()
    
    print("\n" + "="*70)
    print("PREDICTING TEST SET")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_file}\n")
    
    # Load test data
    print("Loading test data...")
    _, _, test_loader = get_data_loaders(
        data_dir=config.DATA_DIR,
        train_label_file=config.TRAIN_LABEL,
        test_label_file=config.TEST_LABEL,
        batch_size=config.BATCH_SIZE,
        val_ratio=config.VAL_RATIO,
        num_workers=config.NUM_WORKERS,
        seed=config.SEED
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Test batches: {len(test_loader)}\n")
    
    # Load model based on model_type
    print("Loading model...")
    
    # Auto-detect model type from checkpoint name if not specified
    if model_type == 'multimodal':
        if 'text_only' in checkpoint_path:
            model_type = 'text_only'
        elif 'image_only' in checkpoint_path:
            model_type = 'image_only'
    
    if model_type == 'text_only':
        model = TextOnlyClassifier(
            num_classes=config.NUM_CLASSES,
            text_model=config.TEXT_MODEL,
            feature_dim=config.FEATURE_DIM,
            dropout=config.DROPOUT
        )
        print("✓ Loaded Text-Only model")
    elif model_type == 'image_only':
        model = ImageOnlyClassifier(
            num_classes=config.NUM_CLASSES,
            image_model=config.IMAGE_MODEL,
            feature_dim=config.FEATURE_DIM,
            dropout=config.DROPOUT
        )
        print("✓ Loaded Image-Only model")
    else:
        model = MultimodalClassifier(
            num_classes=config.NUM_CLASSES,
            text_model=config.TEXT_MODEL,
            image_model=config.IMAGE_MODEL,
            fusion_type=config.FUSION_TYPE,
            feature_dim=config.FEATURE_DIM,
            freeze_encoders=config.FREEZE_ENCODERS,
            dropout=config.DROPOUT
        )
        print(f"✓ Loaded Multimodal model with {config.FUSION_TYPE} fusion")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    print(f"✓ Best accuracy: {checkpoint['best_acc']:.4f}\n")
    
    # Label mapping
    label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    
    # Predict
    print("Predicting...")
    predictions = []
    guids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            texts = batch['text']
            images = batch['image'].to(config.DEVICE)
            batch_guids = batch['guid']
            
            # Forward pass based on model type
            if model_type == 'text_only':
                logits = model(texts)
            elif model_type == 'image_only':
                logits = model(images)
            else:
                logits = model(texts, images)
            preds = torch.argmax(logits, dim=1)
            
            # Collect predictions
            for guid, pred in zip(batch_guids, preds.cpu().numpy()):
                guids.append(guid)
                predictions.append(label_map[pred])
    
    # Save predictions
    print(f"\nSaving predictions to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("guid,tag\n")
        for guid, pred in zip(guids, predictions):
            f.write(f"{guid},{pred}\n")
    
    # Print statistics
    pred_counts = pd.Series(predictions).value_counts()
    print("\nPrediction distribution:")
    for label, count in pred_counts.items():
        print(f"  {label}: {count} ({count/len(predictions)*100:.2f}%)")
    
    print(f"\n✓ Predictions saved to: {output_file}")
    print("="*70 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict test set')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str,
                       default='predictions.txt',
                       help='Output file path')
    parser.add_argument('--model-type', type=str,
                       default='multimodal',
                       choices=['multimodal', 'text_only', 'image_only'],
                       help='Model type (auto-detected from checkpoint name)')
    
    args = parser.parse_args()
    
    predict_test_set(args.checkpoint, args.output, args.model_type)
