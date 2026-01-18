"""
Memory optimization utilities
"""

import torch
import gc


def clear_memory():
    """Clear GPU and CPU memory cache"""
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
        return "CPU mode"


def enable_gradient_checkpointing(model):
    """
    Enable gradient checkpointing to save memory during training
    Trade computation for memory
    """
    if hasattr(model.text_encoder.bert, 'gradient_checkpointing_enable'):
        model.text_encoder.bert.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled for text encoder")
    
    return model


class MemoryEfficientLoader:
    """
    Wrapper for DataLoader with memory management
    """
    
    def __init__(self, dataloader):
        self.dataloader = dataloader
    
    def __iter__(self):
        for batch in self.dataloader:
            yield batch
            # Clear cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def __len__(self):
        return len(self.dataloader)


def optimize_model_memory(model):
    """
    Apply memory optimizations to model
    
    Args:
        model: MultimodalClassifier instance
    
    Returns:
        Optimized model
    """
    # Enable gradient checkpointing
    model = enable_gradient_checkpointing(model)
    
    # Set model to use less memory during eval
    model.eval()
    
    print("✓ Memory optimizations applied")
    return model
