"""
Training utilities with memory optimization
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import gc


class MemoryEfficientTrainer:
    """
    Memory-efficient trainer with gradient accumulation and mixed precision
    """
    
    def __init__(self, 
                 model, 
                 optimizer,
                 device='cpu',
                 accumulation_steps=4,
                 use_amp=False):
        """
        Args:
            model: The model to train
            optimizer: Optimizer
            device: 'cpu' or 'cuda'
            accumulation_steps: Number of steps to accumulate gradients
            use_amp: Use automatic mixed precision (if GPU available)
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp and torch.cuda.is_available()
        
        if self.use_amp:
            self.scaler = GradScaler()
            print("✓ Mixed precision training enabled")
        
        print(f"✓ Gradient accumulation: {accumulation_steps} steps")
        print(f"✓ Effective batch size: batch_size × {accumulation_steps}")
    
    def train_step(self, batch, criterion, step):
        """
        Single training step with gradient accumulation
        
        Args:
            batch: Data batch
            criterion: Loss function
            step: Current step number
            
        Returns:
            loss: Scalar loss value
        """
        texts = batch['text']
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)
        
        # Forward pass with optional mixed precision
        if self.use_amp:
            with autocast():
                logits = self.model(texts, images)
                loss = criterion(logits, labels)
                loss = loss / self.accumulation_steps  # Scale loss
        else:
            logits = self.model(texts, images)
            loss = criterion(logits, labels)
            loss = loss / self.accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every accumulation_steps
        if (step + 1) % self.accumulation_steps == 0:
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Clear cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return loss.item() * self.accumulation_steps  # Unscale for logging
    
    def clear_memory(self):
        """Clear memory cache"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    else:
        import psutil
        process = psutil.Process()
        mem = process.memory_info().rss / 1024**3
        return f"CPU: {mem:.2f}GB RAM used"


def optimize_model_for_training(model, enable_checkpointing=True):
    """
    Apply memory optimizations for training
    
    Args:
        model: MultimodalClassifier instance
        enable_checkpointing: Enable gradient checkpointing
    
    Returns:
        Optimized model
    """
    if enable_checkpointing:
        model.enable_gradient_checkpointing()
    
    # Ensure model is in train mode
    model.train()
    
    # Only trainable params require grad
    for param in model.parameters():
        if not param.requires_grad:
            param.grad = None
    
    print("✓ Model optimized for training")
    return model


# Training configuration recommendations
TRAINING_CONFIG = {
    'batch_size': 4,  # Very small for memory
    'accumulation_steps': 8,  # Effective batch size = 4 × 8 = 32
    'learning_rate': 2e-5,
    'max_epochs': 10,
    'warmup_steps': 100,
    'gradient_clip': 1.0,
    'use_amp': False,  # Set True if GPU available
    'num_workers': 0,  # Windows: must be 0
}
