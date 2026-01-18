"""
Fusion strategies for combining text and image features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusion(nn.Module):
    """
    Late Fusion: Simple concatenation of text and image features
    Most straightforward approach for baseline model
    """
    
    def __init__(self, text_dim=512, image_dim=512):
        """
        Args:
            text_dim: Dimension of text features
            image_dim: Dimension of image features
        """
        super(LateFusion, self).__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.output_dim = text_dim + image_dim
        
        print(f"  ✓ Late Fusion: concat({text_dim} + {image_dim}) = {self.output_dim}")
    
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: [batch_size, text_dim]
            image_features: [batch_size, image_dim]
            
        Returns:
            fused_features: [batch_size, text_dim + image_dim]
        """
        # Simple concatenation
        fused = torch.cat([text_features, image_features], dim=1)
        return fused


class EarlyFusion(nn.Module):
    """
    Early Fusion: Element-wise operations + projection
    For Stage 3 (advanced fusion strategies)
    """
    
    def __init__(self, text_dim=512, image_dim=512, output_dim=512):
        """
        Args:
            text_dim: Dimension of text features
            image_dim: Dimension of image features
            output_dim: Output dimension after fusion
        """
        super(EarlyFusion, self).__init__()
        
        # Align dimensions
        self.text_proj = nn.Linear(text_dim, output_dim)
        self.image_proj = nn.Linear(image_dim, output_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.output_dim = output_dim
        print(f"  ✓ Early Fusion: {text_dim},{image_dim} -> {output_dim}")
    
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: [batch_size, text_dim]
            image_features: [batch_size, image_dim]
            
        Returns:
            fused_features: [batch_size, output_dim]
        """
        # Project to same dimension
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features)
        
        # Element-wise addition
        added = text_proj + image_proj
        
        # Concatenate and fuse
        concat = torch.cat([text_proj, image_proj], dim=1)
        fused = self.fusion(concat)
        
        return fused


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion: Text attends to image and vice versa
    For Stage 3 (advanced fusion strategies)
    """
    
    def __init__(self, text_dim=512, image_dim=512, hidden_dim=256):
        """
        Args:
            text_dim: Dimension of text features
            image_dim: Dimension of image features
            hidden_dim: Hidden dimension for attention
        """
        super(CrossAttentionFusion, self).__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        
        # Text-to-Image attention
        self.text_query = nn.Linear(text_dim, hidden_dim)
        self.image_key = nn.Linear(image_dim, hidden_dim)
        self.image_value = nn.Linear(image_dim, hidden_dim)
        
        # Image-to-Text attention
        self.image_query = nn.Linear(image_dim, hidden_dim)
        self.text_key = nn.Linear(text_dim, hidden_dim)
        self.text_value = nn.Linear(text_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_dim = hidden_dim
        
        print(f"  ✓ Cross-Attention Fusion: {text_dim},{image_dim} -> {hidden_dim}")
    
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: [batch_size, text_dim]
            image_features: [batch_size, image_dim]
            
        Returns:
            fused_features: [batch_size, hidden_dim]
        """
        batch_size = text_features.size(0)
        
        # Text attends to image
        text_q = self.text_query(text_features).unsqueeze(1)  # [B, 1, H]
        image_k = self.image_key(image_features).unsqueeze(1)  # [B, 1, H]
        image_v = self.image_value(image_features).unsqueeze(1)  # [B, 1, H]
        
        # Attention scores
        text_attn_scores = torch.bmm(text_q, image_k.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        text_attn_weights = F.softmax(text_attn_scores, dim=-1)
        text_attended = torch.bmm(text_attn_weights, image_v).squeeze(1)  # [B, H]
        
        # Image attends to text
        image_q = self.image_query(image_features).unsqueeze(1)
        text_k = self.text_key(text_features).unsqueeze(1)
        text_v = self.text_value(text_features).unsqueeze(1)
        
        image_attn_scores = torch.bmm(image_q, text_k.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        image_attn_weights = F.softmax(image_attn_scores, dim=-1)
        image_attended = torch.bmm(image_attn_weights, text_v).squeeze(1)  # [B, H]
        
        # Combine attended features
        fused = torch.cat([text_attended, image_attended], dim=1)
        fused = self.output_proj(fused)
        
        return fused


def test_fusion_modules():
    """Test all fusion strategies"""
    print("\n=== Testing Fusion Modules ===\n")
    
    batch_size = 4
    text_dim = 512
    image_dim = 512
    
    text_features = torch.randn(batch_size, text_dim)
    image_features = torch.randn(batch_size, image_dim)
    
    # Test Late Fusion
    print("[1] Late Fusion")
    late_fusion = LateFusion(text_dim, image_dim)
    late_out = late_fusion(text_features, image_features)
    print(f"    Output shape: {late_out.shape}\n")
    
    # Test Early Fusion
    print("[2] Early Fusion")
    early_fusion = EarlyFusion(text_dim, image_dim, output_dim=256)
    early_out = early_fusion(text_features, image_features)
    print(f"    Output shape: {early_out.shape}\n")
    
    # Test Cross-Attention Fusion
    print("[3] Cross-Attention Fusion")
    cross_attn = CrossAttentionFusion(text_dim, image_dim, hidden_dim=256)
    cross_out = cross_attn(text_features, image_features)
    print(f"    Output shape: {cross_out.shape}\n")
    
    print("✓ All fusion modules test passed!\n")


if __name__ == "__main__":
    test_fusion_modules()
