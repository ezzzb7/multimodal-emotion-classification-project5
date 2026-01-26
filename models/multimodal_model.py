"""
Complete Multimodal Sentiment Classification Model
Combines text and image encoders with fusion strategy
"""

import torch
import torch.nn as nn

from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder
from .fusion import (LateFusion, EarlyFusion, CrossAttentionFusion, 
                     GatedFusion, MultiHeadCrossAttentionFusion, TransformerFusion)
from .advanced_fusion import AlignedFusion, ContrastiveAlignedFusion, HierarchicalFusion


class MultimodalClassifier(nn.Module):
    """
    Complete multimodal model for sentiment classification
    """
    
    def __init__(self,
                 num_classes=3,
                 text_model='distilbert-base-uncased',
                 image_model='resnet50',
                 fusion_type='late',
                 feature_dim=512,
                 freeze_encoders=True,
                 dropout=0.3,
                 transformer_heads=8,
                 transformer_layers=2,
                 transformer_dropout=0.1,
                 transformer_ffn_dim=None,
                 unfreeze_layers=0):
        """
        Args:
            num_classes: Number of sentiment classes (3: pos/neu/neg)
            text_model: Text encoder model name
            image_model: Image encoder model name
            fusion_type: 'late', 'early', 'cross_attention', or 'transformer'
            feature_dim: Feature dimension for encoders
            freeze_encoders: Whether to freeze pretrained encoders
            dropout: Dropout rate for classifier
            transformer_heads: Number of attention heads for transformer fusion
            transformer_layers: Number of transformer layers
            transformer_dropout: Dropout rate for transformer
            transformer_ffn_dim: FFN dimension for transformer (default: feature_dim * 4)
            unfreeze_layers: Number of layers to unfreeze (0=all frozen)
        """
        super(MultimodalClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        
        print(f"\n{'='*60}")
        print(f"Initializing Multimodal Classifier")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  - Num classes: {num_classes}")
        print(f"  - Text model: {text_model}")
        print(f"  - Image model: {image_model}")
        print(f"  - Fusion type: {fusion_type}")
        print(f"  - Feature dim: {feature_dim}")
        print(f"  - Freeze encoders: {freeze_encoders}")
        print(f"  - Dropout: {dropout}")
        print()
        
        # Text encoder
        self.text_encoder = TextEncoder(
            model_name=text_model,
            output_dim=feature_dim,
            freeze_bert=freeze_encoders
        )
        
        # Image encoder
        self.image_encoder = ImageEncoder(
            model_name=image_model,
            output_dim=feature_dim,
            freeze_backbone=freeze_encoders,
            pretrained=True
        )
        
        # Fusion module
        if fusion_type == 'late':
            self.fusion = LateFusion(
                text_dim=feature_dim,
                image_dim=feature_dim
            )
            fusion_dim = feature_dim * 2
        elif fusion_type == 'early':
            self.fusion = EarlyFusion(
                text_dim=feature_dim,
                image_dim=feature_dim,
                output_dim=feature_dim
            )
            fusion_dim = feature_dim
        elif fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(
                text_dim=feature_dim,
                image_dim=feature_dim,
                hidden_dim=feature_dim // 2
            )
            fusion_dim = feature_dim // 2
        elif fusion_type == 'gated':
            # V2新增: 门控融合
            self.fusion = GatedFusion(
                text_dim=feature_dim,
                image_dim=feature_dim,
                output_dim=feature_dim
            )
            fusion_dim = feature_dim
        elif fusion_type == 'multihead_cross_attention':
            # V2新增: 多头交叉注意力融合
            self.fusion = MultiHeadCrossAttentionFusion(
                text_dim=feature_dim,
                image_dim=feature_dim,
                hidden_dim=feature_dim // 2,
                num_heads=4
            )
            fusion_dim = feature_dim // 2
        elif fusion_type == 'transformer':
            # V3新增: Transformer融合（支持配置参数）
            if transformer_ffn_dim is None:
                transformer_ffn_dim = feature_dim * 4  # 默认值
            
            self.fusion = TransformerFusion(
                text_dim=feature_dim,
                image_dim=feature_dim,
                hidden_dim=feature_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=transformer_dropout,
                ffn_dim=transformer_ffn_dim  # ✅ 传递FFN维度配置
            )
            fusion_dim = feature_dim
        elif fusion_type == 'aligned':
            # 高级融合：模态对齐 + 跨模态Transformer
            self.fusion = AlignedFusion(
                text_dim=feature_dim,
                image_dim=feature_dim,
                common_dim=feature_dim,
                num_transformer_layers=transformer_layers,
                nhead=transformer_heads,
                dropout=transformer_dropout,
                hidden_dim=384  # 平衡版：384维（~280万参数）
            )
            fusion_dim = feature_dim
        elif fusion_type == 'contrastive_aligned':
            # 高级融合：对比学习对齐 + 跨模态Transformer
            self.fusion = ContrastiveAlignedFusion(
                text_dim=feature_dim,
                image_dim=feature_dim,
                common_dim=feature_dim,
                num_transformer_layers=transformer_layers,
                nhead=transformer_heads,
                dropout=transformer_dropout
            )
            fusion_dim = feature_dim
            self._use_contrastive_loss = True
        elif fusion_type == 'hierarchical':
            # 高级融合：层次化多级融合
            self.fusion = HierarchicalFusion(
                text_dim=feature_dim,
                image_dim=feature_dim,
                hidden_dim=feature_dim // 2,
                dropout=dropout
            )
            fusion_dim = 128  # 轻量版固定输出128
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Classification head - keep original simple structure
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        print(f"\nClassifier head: {fusion_dim} -> 256 -> {num_classes}")
        print(f"{'='*60}\n")
    
    def forward(self, texts, images, return_features=False):
        """
        Forward pass
        
        Args:
            texts: List of text strings or tokenized dict
            images: Image tensors [batch_size, 3, H, W]
            return_features: Whether to return intermediate features
            
        Returns:
            logits: [batch_size, num_classes]
            (optional) features dict if return_features=True
        """
        # Extract features from each modality
        text_features = self.text_encoder(texts)      # [B, feature_dim]
        image_features = self.image_encoder(images)   # [B, feature_dim]
        
        # Fuse features
        fused_features = self.fusion(text_features, image_features)  # [B, fusion_dim]
        
        # Classify
        logits = self.classifier(fused_features)  # [B, num_classes]
        
        if return_features:
            return logits, {
                'text_features': text_features,
                'image_features': image_features,
                'fused_features': fused_features
            }
        
        return logits
    
    def predict(self, texts, images):
        """
        Predict class labels
        
        Args:
            texts: List of text strings
            images: Image tensors
            
        Returns:
            predictions: Predicted class indices
            probabilities: Class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(texts, images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions, probabilities
    
    def unfreeze_encoders(self, num_layers=2):
        """Unfreeze top layers of encoders for fine-tuning"""
        print("\nUnfreezing encoder layers for fine-tuning...")
        self.text_encoder.unfreeze_layers(num_layers)
        self.image_encoder.unfreeze_layers(num_layers)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        if hasattr(self.text_encoder.bert, 'gradient_checkpointing_enable'):
            self.text_encoder.bert.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled for text encoder")
        return self


class TextOnlyClassifier(nn.Module):
    """
    Text-only model for ablation study (memory optimized)
    """
    
    def __init__(self,
                 num_classes=3,
                 text_model='distilbert-base-uncased',
                 feature_dim=512,
                 dropout=0.3):
        super(TextOnlyClassifier, self).__init__()
        
        self.text_encoder = TextEncoder(
            model_name=text_model,
            output_dim=feature_dim,
            freeze_bert=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        print(f"Text-only classifier: {feature_dim} -> 128 -> {num_classes}")
    
    def forward(self, texts):
        text_features = self.text_encoder(texts)
        logits = self.classifier(text_features)
        return logits


class ImageOnlyClassifier(nn.Module):
    """
    Image-only model for ablation study (memory optimized)
    """
    
    def __init__(self,
                 num_classes=3,
                 image_model='resnet50',
                 feature_dim=512,
                 dropout=0.3):
        super(ImageOnlyClassifier, self).__init__()
        
        self.image_encoder = ImageEncoder(
            model_name=image_model,
            output_dim=feature_dim,
            freeze_backbone=True,
            pretrained=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),  # Reduced
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        print(f"Image-only classifier: {feature_dim} -> 128 -> {num_classes}")
    
    def forward(self, images):
        image_features = self.image_encoder(images)
        logits = self.classifier(image_features)
        return logits


def test_multimodal_model():
    """Test multimodal model"""
    print("\n=== Testing Multimodal Model ===\n")
    
    # Create model
    model = MultimodalClassifier(
        num_classes=3,
        fusion_type='late',
        feature_dim=256,  # Smaller for testing
        freeze_encoders=True
    )
    
    # Test inputs
    batch_size = 2
    texts = [
        "I love this beautiful sunset!",
        "This is terrible and disappointing."
    ]
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        logits = model(texts, images)
    
    print(f"Input: {batch_size} samples")
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits: {logits}")
    
    # Test prediction
    predictions, probabilities = model.predict(texts, images)
    print(f"\nPredictions: {predictions}")
    print(f"Probabilities:\n{probabilities}")
    
    print("\n✓ Multimodal model test passed!\n")


if __name__ == "__main__":
    test_multimodal_model()
