"""
改进的模型架构
1. 注意力机制融合
2. 门控融合机制
3. 多头注意力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """注意力机制融合"""
    
    def __init__(self, text_dim, image_dim, hidden_dim=256):
        super().__init__()
        
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        
        # 注意力权重计算
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),  # 2个模态的权重
            nn.Softmax(dim=-1)
        )
        
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: [batch, text_dim]
            image_features: [batch, image_dim]
        Returns:
            fused_features: [batch, hidden_dim]
        """
        # 投影到相同维度
        text_proj = self.text_proj(text_features)  # [B, hidden_dim]
        image_proj = self.image_proj(image_features)  # [B, hidden_dim]
        
        # 计算注意力权重
        concat_features = torch.cat([text_proj, image_proj], dim=1)  # [B, hidden_dim*2]
        attention_weights = self.attention(concat_features)  # [B, 2]
        
        # 加权融合
        text_weighted = text_proj * attention_weights[:, 0:1]  # [B, hidden_dim]
        image_weighted = image_proj * attention_weights[:, 1:2]  # [B, hidden_dim]
        
        # 最终融合
        fused = torch.cat([text_weighted, image_weighted], dim=1)
        fused = self.fusion(fused)
        
        return fused


class GatedFusion(nn.Module):
    """门控融合机制"""
    
    def __init__(self, text_dim, image_dim, hidden_dim=512):
        super().__init__()
        
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        
        # 门控单元
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: [batch, text_dim]
            image_features: [batch, image_dim]
        Returns:
            fused_features: [batch, hidden_dim]
        """
        # 投影
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features)
        
        # 门控权重
        concat = torch.cat([text_proj, image_proj], dim=1)
        gate_weight = self.gate(concat)
        
        # 门控融合
        fused = gate_weight * text_proj + (1 - gate_weight) * image_proj
        fused = self.output_proj(fused)
        
        return fused


class MultiHeadAttentionFusion(nn.Module):
    """多头注意力融合"""
    
    def __init__(self, text_dim, image_dim, hidden_dim=512, num_heads=4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # 投影层
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        
        # Q, K, V投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, text_features, image_features):
        """
        Args:
            text_features: [batch, text_dim]
            image_features: [batch, image_dim]
        Returns:
            fused_features: [batch, hidden_dim]
        """
        batch_size = text_features.size(0)
        
        # 投影到相同维度
        text_proj = self.text_proj(text_features).unsqueeze(1)  # [B, 1, hidden_dim]
        image_proj = self.image_proj(image_features).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # 拼接作为序列
        features = torch.cat([text_proj, image_proj], dim=1)  # [B, 2, hidden_dim]
        
        # Q, K, V
        Q = self.q_proj(features)  # [B, 2, hidden_dim]
        K = self.k_proj(features)
        V = self.v_proj(features)
        
        # 重塑为多头
        Q = Q.view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, 2, head_dim]
        K = K.view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 2, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, num_heads, 2, 2]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        attended = torch.matmul(attn, V)  # [B, num_heads, 2, head_dim]
        
        # 合并多头
        attended = attended.transpose(1, 2).contiguous().view(batch_size, 2, self.hidden_dim)
        
        # 输出投影
        output = self.out_proj(attended)  # [B, 2, hidden_dim]
        
        # 池化为单个向量
        fused = output.mean(dim=1)  # [B, hidden_dim]
        
        return fused


class ImprovedMultimodalClassifier(nn.Module):
    """改进的多模态分类器（可选融合策略）"""
    
    def __init__(self, num_classes=3, text_model='distilbert-base-uncased',
                 image_model='resnet50', fusion_type='attention',
                 text_dim=768, image_dim=2048, hidden_dim=512,
                 freeze_encoders=True, dropout=0.3):
        super().__init__()
        
        from models.text_encoder import TextEncoder
        from models.image_encoder import ImageEncoder
        
        # 编码器
        self.text_encoder = TextEncoder(text_model, text_dim, freeze_bert=freeze_encoders)
        self.image_encoder = ImageEncoder(image_model, image_dim, freeze_backbone=freeze_encoders)
        
        # 融合层
        if fusion_type == 'attention':
            self.fusion = AttentionFusion(text_dim, image_dim, hidden_dim)
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(text_dim, image_dim, hidden_dim)
        elif fusion_type == 'multihead':
            self.fusion = MultiHeadAttentionFusion(text_dim, image_dim, hidden_dim, num_heads=4)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        print(f"✓ Improved Multimodal Classifier with {fusion_type} fusion")
    
    def forward(self, texts, images):
        """
        Args:
            texts: list of text strings or tokenized dict
            images: [batch, 3, H, W]
        Returns:
            logits: [batch, num_classes]
        """
        text_features = self.text_encoder(texts)
        image_features = self.image_encoder(images)
        
        fused_features = self.fusion(text_features, image_features)
        
        logits = self.classifier(fused_features)
        
        return logits


def test_fusion_modules():
    """测试融合模块"""
    batch_size = 4
    text_dim = 768
    image_dim = 2048
    hidden_dim = 512
    
    text_features = torch.randn(batch_size, text_dim)
    image_features = torch.randn(batch_size, image_dim)
    
    print("测试融合模块:")
    print("="*50)
    
    # 1. 注意力融合
    attention_fusion = AttentionFusion(text_dim, image_dim, hidden_dim)
    output = attention_fusion(text_features, image_features)
    print(f"✓ Attention Fusion: {output.shape}")
    
    # 2. 门控融合
    gated_fusion = GatedFusion(text_dim, image_dim, hidden_dim)
    output = gated_fusion(text_features, image_features)
    print(f"✓ Gated Fusion: {output.shape}")
    
    # 3. 多头注意力融合
    multihead_fusion = MultiHeadAttentionFusion(text_dim, image_dim, hidden_dim, num_heads=4)
    output = multihead_fusion(text_features, image_features)
    print(f"✓ MultiHead Fusion: {output.shape}")
    
    print("="*50)
    print("所有模块测试通过！")


if __name__ == '__main__':
    test_fusion_modules()
