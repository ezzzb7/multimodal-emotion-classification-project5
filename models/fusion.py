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


class TransformerFusion(nn.Module):
    """
    Transformer-Based Cross-Modal Fusion (V3)
    使用Multi-Head Self-Attention对齐和融合跨模态特征
    
    参考论文：
    - ViLBERT (NeurIPS 2019): Co-attentional Transformer
    - UNITER (ECCV 2020): Universal Image-Text Representation
    """
    
    def __init__(self, text_dim=768, image_dim=2048, hidden_dim=512, 
                 num_heads=8, num_layers=2, dropout=0.1, ffn_dim=None):
        """
        Args:
            text_dim: Text feature dimension (DistilBERT: 768)
            image_dim: Image feature dimension (ResNet50: 2048)
            hidden_dim: Hidden dimension for transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            dropout: Dropout rate
            ffn_dim: FFN dimension (default: hidden_dim * 4)
        """
        super(TransformerFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 设置FFN维度（支持配置）
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4
        self.ffn_dim = ffn_dim
        
        # 投影到统一的隐藏空间
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 模态类型嵌入（区分text和image）
        self.text_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.image_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # 位置嵌入（简单的可学习嵌入）
        self.pos_embedding = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,  # ✅ 使用可配置的FFN维度
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LayerNorm（更稳定）
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        self.output_dim = hidden_dim
        
        # 初始化
        self._init_weights()
        
        print(f"  ✓ Transformer Fusion: {text_dim},{image_dim} -> {hidden_dim}")
        print(f"    - Heads: {num_heads}, Layers: {num_layers}")
        print(f"    - FFN dim: {ffn_dim}, Dropout: {dropout}")  # ✅ 显示实际FFN维度
    
    def _init_weights(self):
        """Xavier初始化"""
        nn.init.xavier_uniform_(self.text_type_embedding)
        nn.init.xavier_uniform_(self.image_type_embedding)
        nn.init.xavier_uniform_(self.pos_embedding)
    
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: [batch_size, text_dim]
            image_features: [batch_size, image_dim]
            
        Returns:
            fused_features: [batch_size, hidden_dim]
        """
        batch_size = text_features.size(0)
        
        # 投影到统一空间
        text_proj = self.text_proj(text_features).unsqueeze(1)  # [B, 1, D]
        image_proj = self.image_proj(image_features).unsqueeze(1)  # [B, 1, D]
        
        # 添加模态类型嵌入
        text_proj = text_proj + self.text_type_embedding
        image_proj = image_proj + self.image_type_embedding
        
        # 拼接成序列 [B, 2, D]
        sequence = torch.cat([text_proj, image_proj], dim=1)
        
        # 添加位置嵌入
        sequence = sequence + self.pos_embedding
        
        # Transformer处理（自注意力实现跨模态交互）
        # [B, 2, D] -> [B, 2, D]
        transformed = self.transformer_encoder(sequence)
        
        # 池化策略：拼接两个模态的表示
        # 也可以尝试：平均池化、加权池化、只取CLS token等
        pooled = transformed.reshape(batch_size, -1)  # [B, 2*D]
        
        # 输出投影
        output = self.output_proj(pooled)  # [B, D]
        
        return output


class GatedFusion(nn.Module):
    """
    Gated Fusion: Learn dynamic weights for each modality
    门控融合：自动学习每个模态的重要性权重
    """
    
    def __init__(self, text_dim=512, image_dim=512, output_dim=512):
        """
        Args:
            text_dim: Dimension of text features
            image_dim: Dimension of image features
            output_dim: Output dimension
        """
        super(GatedFusion, self).__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.output_dim = output_dim
        
        # 门控网络 - 学习每个模态的重要性
        self.gate = nn.Sequential(
            nn.Linear(text_dim + image_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)  # 输出 [gate_text, gate_image]
        )
        
        # 对齐投影
        self.text_proj = nn.Linear(text_dim, output_dim)
        self.image_proj = nn.Linear(image_dim, output_dim)
        
        # 融合投影
        self.fusion_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        print(f"  ✓ Gated Fusion: {text_dim},{image_dim} -> {output_dim}")
    
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: [batch_size, text_dim]
            image_features: [batch_size, image_dim]
            
        Returns:
            fused_features: [batch_size, output_dim]
        """
        # 计算门控权重
        combined = torch.cat([text_features, image_features], dim=1)
        gates = self.gate(combined)  # [B, 2]
        gate_text = gates[:, 0:1]  # [B, 1]
        gate_image = gates[:, 1:2]  # [B, 1]
        
        # 投影到相同维度
        text_proj = self.text_proj(text_features)  # [B, output_dim]
        image_proj = self.image_proj(image_features)  # [B, output_dim]
        
        # 加权融合
        fused = gate_text * text_proj + gate_image * image_proj
        
        # 最终投影
        fused = self.fusion_proj(fused)
        
        return fused


class MultiHeadCrossAttentionFusion(nn.Module):
    """
    Multi-Head Cross-Attention Fusion
    多头交叉注意力融合：更强大的注意力机制
    """
    
    def __init__(self, text_dim=512, image_dim=512, hidden_dim=256, num_heads=4):
        """
        Args:
            text_dim: Dimension of text features
            image_dim: Dimension of image features
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
        """
        super(MultiHeadCrossAttentionFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 投影到相同维度
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.output_dim = hidden_dim
        print(f"  ✓ Multi-Head Cross-Attention Fusion: {text_dim},{image_dim} -> {hidden_dim} (heads={num_heads})")
    
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: [batch_size, text_dim]
            image_features: [batch_size, image_dim]
            
        Returns:
            fused_features: [batch_size, hidden_dim]
        """
        # 投影
        text_proj = self.text_proj(text_features).unsqueeze(1)  # [B, 1, H]
        image_proj = self.image_proj(image_features).unsqueeze(1)  # [B, 1, H]
        
        # Text attends to image
        text_attended, _ = self.multihead_attn(
            query=text_proj, 
            key=image_proj, 
            value=image_proj
        )  # [B, 1, H]
        
        # Image attends to text
        image_attended, _ = self.multihead_attn(
            query=image_proj, 
            key=text_proj, 
            value=text_proj
        )  # [B, 1, H]
        
        # Squeeze and concat
        text_attended = text_attended.squeeze(1)  # [B, H]
        image_attended = image_attended.squeeze(1)  # [B, H]
        
        # Fuse
        fused = torch.cat([text_attended, image_attended], dim=1)  # [B, 2H]
        fused = self.fusion(fused)  # [B, H]
        
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
    
    # Test Gated Fusion (V2)
    print("[4] Gated Fusion (V2)")
    gated = GatedFusion(text_dim, image_dim, output_dim=256)
    gated_out = gated(text_features, image_features)
    print(f"    Output shape: {gated_out.shape}\n")
    
    # Test Multi-Head Cross-Attention Fusion (V2)
    print("[5] Multi-Head Cross-Attention Fusion (V2)")
    mhca = MultiHeadCrossAttentionFusion(text_dim, image_dim, hidden_dim=256, num_heads=4)
    mhca_out = mhca(text_features, image_features)
    print(f"    Output shape: {mhca_out.shape}\n")
    
    # Test Transformer Fusion (V3)
    print("[6] Transformer Fusion (V3)")
    transformer = TransformerFusion(text_dim, image_dim, hidden_dim=512, num_heads=8, num_layers=2)
    transformer_out = transformer(text_features, image_features)
    print(f"    Output shape: {transformer_out.shape}\n")
    
    print("✓ All fusion modules test passed!\n")


if __name__ == "__main__":
    test_fusion_modules()
