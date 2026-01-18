"""
Complete Multimodal Sentiment Classification Model
Combines text and image encoders with fusion strategy
"""

import torch
import torch.nn as nn

from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder
from .fusion import LateFusion, EarlyFusion, CrossAttentionFusion


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
                 dropout=0.3):
        """
        Args:
            num_classes: Number of sentiment classes (3: pos/neu/neg)
            text_model: Text encoder model name
            image_model: Image encoder model name
            fusion_type: 'late', 'early', or 'cross_attention'
            feature_dim: Feature dimension for encoders
            freeze_encoders: Whether to freeze pretrained encoders
            dropout: Dropout rate for classifier
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
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Classification head (memory optimized)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),  # Reduced from 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)  # Direct to output
        )
        
        print(f"\nClassifier head: {fusion_dim} -> 128 -> {num_classes}")
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
