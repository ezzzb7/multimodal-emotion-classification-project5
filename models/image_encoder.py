"""
Image Encoder using pretrained ResNet50
Extracts visual features from images
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """
    Image encoder using pretrained ResNet50
    Extracts visual features from image input
    """
    
    def __init__(self,
                 model_name='resnet50',
                 output_dim=512,
                 freeze_backbone=True,
                 pretrained=True):
        """
        Args:
            model_name: Backbone model name (resnet50, resnet34, etc.)
            output_dim: Output feature dimension
            freeze_backbone: Whether to freeze backbone parameters
            pretrained: Use ImageNet pretrained weights
        """
        super(ImageEncoder, self).__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        
        # Load pretrained ResNet
        print(f"Loading {model_name} (pretrained={pretrained})...")
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
            print(f"  ResNet50: {self.feature_dim} features")
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Freeze backbone parameters
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"  ✓ Backbone parameters frozen")
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        print(f"  ✓ Image encoder initialized: {self.feature_dim} -> {output_dim}")
    
    def forward(self, images):
        """
        Forward pass
        
        Args:
            images: Tensor of shape [batch_size, 3, H, W]
            
        Returns:
            torch.Tensor: Image features [batch_size, output_dim]
        """
        # Extract features from backbone
        features = self.backbone(images)  # [batch_size, feature_dim, 1, 1]
        
        # Project to output dimension
        image_features = self.projection(features)  # [batch_size, output_dim]
        
        return image_features
    
    def unfreeze_layers(self, num_layers=2):
        """
        Unfreeze top N layers for fine-tuning
        
        Args:
            num_layers: Number of top blocks to unfreeze
        """
        # Get all sequential modules
        modules = list(self.backbone.children())
        
        # Unfreeze last N modules
        for module in modules[-num_layers:]:
            for param in module.parameters():
                param.requires_grad = True
        
        print(f"  ✓ Unfroze top {num_layers} backbone layers")


def test_image_encoder():
    """Test function"""
    print("\n=== Testing Image Encoder ===")
    
    # Create encoder
    encoder = ImageEncoder(
        model_name='resnet50',
        output_dim=512,
        freeze_backbone=True,
        pretrained=True
    )
    
    # Test images (random tensors)
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        features = encoder(images)
    
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Feature sample: {features[0][:5]}")
    print("✓ Image encoder test passed!\n")


if __name__ == "__main__":
    test_image_encoder()
