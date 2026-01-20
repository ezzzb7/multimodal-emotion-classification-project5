"""
Training script for multimodal sentiment classification
Memory-optimized for limited resources
"""

import sys
import os
import time
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from configs.config import get_config
from data.data_loader import get_data_loaders
from models.multimodal_model import MultimodalClassifier, TextOnlyClassifier, ImageOnlyClassifier
from utils.train_utils import (
    set_seed, count_parameters, save_checkpoint,
    compute_metrics, AverageMeter, EarlyStopping
)
from utils.logger import TrainingLogger


def train_epoch(model, train_loader, criterion, optimizer, scheduler, config, epoch, logger=None, model_type='multimodal'):
    """Train for one epoch"""
    model.train()
    
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
    
    for step, batch in enumerate(pbar):
        texts = batch['text']
        images = batch['image'].to(config.DEVICE)
        labels = batch['label'].to(config.DEVICE)
        
        # Forward pass based on model type
        if model_type == 'text_only':
            logits = model(texts)
        elif model_type == 'image_only':
            logits = model(images)
        else:  # multimodal
            logits = model(texts, images)
        loss = criterion(logits, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / config.ACCUMULATION_STEPS
        loss.backward()
        
        # Clear input tensors immediately
        del texts, images
        
        # Accumulate gradients
        if (step + 1) % config.ACCUMULATION_STEPS == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Metrics
        loss_meter.update(loss.item() * config.ACCUMULATION_STEPS, labels.size(0))
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Clear tensors
        del logits, preds, labels
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
        # Logging
        if (step + 1) % config.LOG_EVERY == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"\n  Step {step+1}: loss={loss_meter.avg:.4f}, lr={lr:.2e}")
            
            # Log to logger (step-level, optional)
            if logger and (step + 1) % config.ACCUMULATION_STEPS == 0:
                logger.log_step(epoch, step + 1, loss_meter.avg, lr)
    
    # Compute epoch metrics
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = loss_meter.avg
    
    return metrics, all_preds, all_labels


def evaluate(model, val_loader, criterion, config, logger=None, model_type='multimodal'):
    """Evaluate on validation set"""
    model.eval()
    
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []
    all_guids = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            texts = batch['text']
            images = batch['image'].to(config.DEVICE)
            labels = batch['label'].to(config.DEVICE)
            guids = batch['guid']  # 获取样本ID
            
            # Forward pass based on model type
            if model_type == 'text_only':
                logits = model(texts)
            elif model_type == 'image_only':
                logits = model(images)
            else:  # multimodal
                logits = model(texts, images)
            loss = criterion(logits, labels)
            
            # Clear unnecessary tensors immediately
            del texts, images
            
            # Metrics
            loss_meter.update(loss.item(), labels.size(0))
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_guids.extend(guids)  # 收集guid
            
            # Clear tensors
            del logits, preds, labels
    
    # Compute metrics
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = loss_meter.avg
    
    # Log error samples
    if logger:
        logger.log_error_samples(all_guids, all_labels, all_preds)
    
    return metrics, all_preds, all_labels


def main():
    """Main training function"""
    
    # Load config
    config = get_config()
    
    # Check for resume
    resume_checkpoint = None
    start_epoch = 0
    if hasattr(config, 'RESUME_FROM') and config.RESUME_FROM:
        if os.path.exists(config.RESUME_FROM):
            print(f"\n🔄 Resuming training from: {config.RESUME_FROM}")
            resume_checkpoint = torch.load(config.RESUME_FROM, map_location=config.DEVICE, weights_only=False)
            start_epoch = resume_checkpoint['epoch'] + 1
            print(f"   Starting from epoch {start_epoch + 1}")
        else:
            print(f"\n⚠ Warning: Resume checkpoint not found: {config.RESUME_FROM}")
            print("   Starting from scratch...")
    
    print("\n" + "="*70)
    print("TRAINING MULTIMODAL SENTIMENT CLASSIFIER")
    print("="*70)
    print(f"Device: {config.DEVICE}")
    print(f"Fusion: {config.FUSION_TYPE}")
    print(f"Batch size: {config.BATCH_SIZE} × {config.ACCUMULATION_STEPS} = {config.BATCH_SIZE * config.ACCUMULATION_STEPS} (effective)")
    print(f"Epochs: {start_epoch + 1}-{config.NUM_EPOCHS}" if start_epoch > 0 else f"Epochs: {config.NUM_EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    if resume_checkpoint:
        print(f"Resume from: Epoch {resume_checkpoint['epoch']}, Best Acc: {resume_checkpoint.get('best_acc', 0):.4f}")
    print("="*70 + "\n")
    
    # Set seed
    set_seed(config.SEED)
    
    # Create directories
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, _ = get_data_loaders(
        data_dir=config.DATA_DIR,
        train_label_file=config.TRAIN_LABEL,
        batch_size=config.BATCH_SIZE,
        val_ratio=config.VAL_RATIO,
        num_workers=config.NUM_WORKERS,
        seed=config.SEED
    )
    
    # Create model based on model_type
    print("\nCreating model...")
    model_type = getattr(config, 'MODEL_TYPE', 'multimodal')
    
    if model_type == 'text_only':
        model = TextOnlyClassifier(
            num_classes=config.NUM_CLASSES,
            text_model=config.TEXT_MODEL,
            feature_dim=config.FEATURE_DIM,
            dropout=config.DROPOUT
        )
        print(f"✓ Created Text-Only model")
    elif model_type == 'image_only':
        model = ImageOnlyClassifier(
            num_classes=config.NUM_CLASSES,
            image_model=config.IMAGE_MODEL,
            feature_dim=config.FEATURE_DIM,
            dropout=config.DROPOUT
        )
        print(f"✓ Created Image-Only model")
    else:  # multimodal
        model = MultimodalClassifier(
            num_classes=config.NUM_CLASSES,
            text_model=config.TEXT_MODEL,
            image_model=config.IMAGE_MODEL,
            fusion_type=config.FUSION_TYPE,
            feature_dim=config.FEATURE_DIM,
            freeze_encoders=config.FREEZE_ENCODERS,
            dropout=config.DROPOUT
        )
        print(f"✓ Created Multimodal model with {config.FUSION_TYPE} fusion")
    
    # Enable gradient checkpointing
    if config.GRADIENT_CHECKPOINTING and hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
        print("✓ Gradient checkpointing enabled")
    
    model = model.to(config.DEVICE)
    
    # Load model weights if resuming
    if resume_checkpoint:
        model.load_state_dict(resume_checkpoint['model_state_dict'])
        print("✓ Model weights loaded from checkpoint")
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Loss and optimizer (with class weights for imbalanced data)
    # Calculate class weights: inverse of class frequency
    train_labels = []
    for batch in train_loader:
        train_labels.extend(batch['label'].numpy())
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(config.DEVICE)
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Layered learning rate optimization
    # If ENCODER_LR is specified, use different learning rates for encoders vs classifier
    if hasattr(config, 'ENCODER_LR') and config.ENCODER_LR is not None:
        print(f"\n⚡ Using layered learning rates:")
        print(f"  - Encoders: {config.ENCODER_LR}")
        print(f"  - Classifier/Fusion: {config.LEARNING_RATE}")
        
        # Separate parameters into groups
        encoder_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'text_encoder' in name or 'image_encoder' in name:
                encoder_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = AdamW([
            {'params': encoder_params, 'lr': config.ENCODER_LR},
            {'params': other_params, 'lr': config.LEARNING_RATE}
        ], weight_decay=config.WEIGHT_DECAY)
        
        print(f"  - Encoder parameters: {sum(p.numel() for p in encoder_params):,}")
        print(f"  - Other parameters: {sum(p.numel() for p in other_params):,}")
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    
    # Load optimizer state if resuming
    if resume_checkpoint and 'optimizer_state_dict' in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        print("✓ Optimizer state loaded from checkpoint")
    
    # Learning rate scheduler with warmup ratio
    total_steps = len(train_loader) * config.NUM_EPOCHS // config.ACCUMULATION_STEPS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    # Initialize best accuracy
    best_acc = resume_checkpoint.get('best_acc', 0.0) if resume_checkpoint else 0.0
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        mode='max'  # Maximize accuracy
    )
    
    # Create logger with meaningful experiment name
    # Format: {fusion_type}_{modality}_{YYYYMMDD_HHMMSS}
    # e.g., late_multimodal_20260118_170456, early_text_only_20260118_180000
    if resume_checkpoint and 'experiment_name' in resume_checkpoint:
        # Reuse the same experiment name when resuming
        experiment_name = resume_checkpoint['experiment_name']
        print(f"✓ Resuming experiment: {experiment_name}")
    elif hasattr(config, 'EXPERIMENT_NAME') and config.EXPERIMENT_NAME:
        experiment_name = config.EXPERIMENT_NAME
    else:
        # Auto-generate based on model type and modality
        modality = getattr(config, 'MODALITY', 'multimodal')
        if model_type == 'multimodal':
            experiment_name = f"{config.FUSION_TYPE}_{modality}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            experiment_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger = TrainingLogger(log_dir=config.LOG_DIR, experiment_name=experiment_name)
    
    # Log configuration
    config_dict = {
        'fusion_type': config.FUSION_TYPE,
        'feature_dim': config.FEATURE_DIM,
        'batch_size': config.BATCH_SIZE,
        'accumulation_steps': config.ACCUMULATION_STEPS,
        'learning_rate': config.LEARNING_RATE,
        'num_epochs': config.NUM_EPOCHS,
        'dropout': config.DROPOUT,
        'freeze_encoders': config.FREEZE_ENCODERS,
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    logger.log_config(config_dict)
    
    # Training loop
    train_start = time.time()
    
    print("\n" + "="*70)
    print("STARTING TRAINING" if start_epoch == 0 else f"RESUMING TRAINING (Epoch {start_epoch + 1}-{config.NUM_EPOCHS})")
    print("="*70 + "\n")
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_metrics, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, config, epoch, logger, model_type
        )
        
        print(f"\nEpoch {epoch+1} Training:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Acc: {train_metrics['accuracy']:.4f}")
        print(f"  F1: {train_metrics['f1']:.4f}")
        
        # Evaluate
        if (epoch + 1) % config.EVAL_EVERY == 0:
            val_metrics, val_preds, val_labels = evaluate(
                model, val_loader, criterion, config, logger, model_type
            )
            
            print(f"\nEpoch {epoch+1} Validation:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Precision: {val_metrics['precision']:.4f}")
            print(f"  Recall: {val_metrics['recall']:.4f}")
            print(f"  F1: {val_metrics['f1']:.4f}")
            
            # Log to logger
            epoch_time = time.time() - epoch_start
            lr = optimizer.param_groups[0]['lr']
            logger.log_epoch(epoch + 1, train_metrics, val_metrics, lr, epoch_time)
            
            # Save best model with experiment-specific name
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                # Save to both generic and experiment-specific paths
                best_model_name = f"best_{experiment_name}.pth"
                save_path = os.path.join(config.SAVE_DIR, best_model_name)
                save_checkpoint(model, optimizer, epoch, best_acc, save_path, experiment_name)
                print(f"  ✓ New best accuracy: {best_acc:.4f} -> {best_model_name}")
            
            # Early stopping
            if early_stopping(val_metrics['accuracy'], epoch):
                break
        
        # Save periodic checkpoint with experiment name
        if (epoch + 1) % config.SAVE_EVERY == 0:
            checkpoint_name = f"{experiment_name}_epoch{epoch+1}.pth"
            save_path = os.path.join(config.SAVE_DIR, checkpoint_name)
            save_checkpoint(model, optimizer, epoch, best_acc, save_path, experiment_name)
        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1} completed in {epoch_time/60:.2f} minutes")
        print("="*70 + "\n")
    
    # Training complete
    train_time = time.time() - train_start
    
    # Save final summary
    logger.save_final_summary()
    
    print(f"\nTraining completed in {train_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Model saved to: {config.SAVE_DIR}/best_model.pth")
    
    return logger


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
