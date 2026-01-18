"""
Models module for multimodal sentiment classification
"""

from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder
from .fusion import LateFusion, EarlyFusion, CrossAttentionFusion
from .multimodal_model import MultimodalClassifier

__all__ = [
    'TextEncoder',
    'ImageEncoder',
    'LateFusion',
    'EarlyFusion',
    'CrossAttentionFusion',
    'MultimodalClassifier'
]
