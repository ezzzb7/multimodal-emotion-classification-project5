"""
Evaluation script for trained models
"""

import sys
import os
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from configs.config import get_config
from data.data_loader import get_data_loaders
from models.multimodal_model import MultimodalClassifier
from utils.train_utils import load_checkpoint, compute_metrics, plot_confusion_matrix


def evaluate_model(model, data_loader, config, label_names=['positive', 'neutral', 'negative']):
    """Evaluate model on dataset"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_guids = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            texts = batch['text']
            images = batch['image'].to(config.DEVICE)
            labels = batch['label']
            guids = batch['guid']
            
            # Forward pass
            logits = model(texts, images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_guids.extend(guids)
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_labels)
    
    return metrics, all_preds, all_labels, all_guids


def main():
    """Main evaluation function"""
    
    # Load config
    config = get_config()
    
    print("\n" + "="*70)
    print("EVALUATING TRAINED MODEL")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    _, val_loader, _ = get_data_loaders(
        data_dir=config.DATA_DIR,
        train_label_file=config.TRAIN_LABEL,
        batch_size=config.BATCH_SIZE,
        val_ratio=config.VAL_RATIO,
        num_workers=config.NUM_WORKERS,
        seed=config.SEED
    )
    
    # Create model
    print("\nLoading model...")
    model = MultimodalClassifier(
        num_classes=config.NUM_CLASSES,
        text_model=config.TEXT_MODEL,
        image_model=config.IMAGE_MODEL,
        fusion_type=config.FUSION_TYPE,
        feature_dim=config.FEATURE_DIM,
        freeze_encoders=config.FREEZE_ENCODERS,
        dropout=config.DROPOUT
    )
    
    # Load checkpoint
    checkpoint_path = os.path.join(config.SAVE_DIR, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    model, _, epoch, best_acc = load_checkpoint(model, None, checkpoint_path)
    model = model.to(config.DEVICE)
    
    # Evaluate
    print("\nEvaluating...")
    metrics, preds, labels, guids = evaluate_model(model, val_loader, config)
    
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("="*70 + "\n")
    
    # Plot confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(
        labels, preds,
        classes=['positive', 'neutral', 'negative'],
        save_path=os.path.join(config.LOG_DIR, 'confusion_matrix.png')
    )
    
    # Save predictions
    import pandas as pd
    df = pd.DataFrame({
        'guid': guids,
        'true_label': labels,
        'pred_label': preds
    })
    output_path = os.path.join(config.LOG_DIR, 'val_predictions.csv')
    df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved: {output_path}\n")


if __name__ == "__main__":
    main()
