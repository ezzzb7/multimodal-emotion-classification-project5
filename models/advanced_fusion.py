"""
高级多模态融合模块
解决核心问题：文本和图像表征不在同一语义空间

设计思路：
1. 模态对齐层 (Modality Alignment): 将不同模态投影到共同语义空间
2. 跨模态交互层 (Cross-Modal Interaction): 让两个模态相互"看到"对方
3. 融合层 (Fusion): 在对齐的空间中进行有效融合

参考思想：
- CLIP: 对比学习对齐文本和图像空间
- ViLBERT: 双流Transformer跨模态交互
- UNITER: 统一表征学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ModalityAlignmentLayer(nn.Module):
    """
    模态对齐层：将文本和图像表征投影到共同的语义空间
    
    核心思想：不同模态的原始表征在各自的"语言"中，
    需要先"翻译"到共同语言才能有效融合
    """
    def __init__(self, text_dim: int, image_dim: int, common_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 文本投影：text_dim → common_dim
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim, common_dim),
            nn.LayerNorm(common_dim)
        )
        
        # 图像投影：image_dim → common_dim  
        self.image_projector = nn.Sequential(
            nn.Linear(image_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim, common_dim),
            nn.LayerNorm(common_dim)
        )
        
    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor):
        """
        输入: text_feat [B, text_dim], image_feat [B, image_dim]
        输出: aligned_text [B, common_dim], aligned_image [B, common_dim]
        """
        aligned_text = self.text_projector(text_feat)
        aligned_image = self.image_projector(image_feat)
        return aligned_text, aligned_image


class CrossModalTransformerLayer(nn.Module):
    """
    跨模态Transformer层：让两个模态相互交互
    
    与标准Transformer的区别：
    - Query来自一个模态，Key/Value来自另一个模态
    - 实现双向跨模态注意力
    """
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # 文本作为Query，图像作为Key/Value
        self.text_to_image_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 图像作为Query，文本作为Key/Value
        self.image_to_text_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # FFN for text
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # FFN for image
        self.image_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_norm2 = nn.LayerNorm(d_model)
        self.image_norm1 = nn.LayerNorm(d_model)
        self.image_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor):
        """
        输入: text_feat [B, common_dim], image_feat [B, common_dim]
        输出: enhanced_text [B, common_dim], enhanced_image [B, common_dim]
        """
        # 扩展维度以适配MultiheadAttention: [B, D] -> [B, 1, D]
        text_feat = text_feat.unsqueeze(1)
        image_feat = image_feat.unsqueeze(1)
        
        # 文本关注图像 (Text attends to Image)
        text_attended, _ = self.text_to_image_attn(
            query=text_feat,
            key=image_feat, 
            value=image_feat
        )
        text_feat = self.text_norm1(text_feat + text_attended)
        text_feat = self.text_norm2(text_feat + self.text_ffn(text_feat))
        
        # 图像关注文本 (Image attends to Text)
        image_attended, _ = self.image_to_text_attn(
            query=image_feat,
            key=text_feat,
            value=text_feat
        )
        image_feat = self.image_norm1(image_feat + image_attended)
        image_feat = self.image_norm2(image_feat + self.image_ffn(image_feat))
        
        # 压缩回原维度: [B, 1, D] -> [B, D]
        return text_feat.squeeze(1), image_feat.squeeze(1)


class AlignedFusion(nn.Module):
    """
    对齐融合 (Aligned Fusion) - 可配置版本
    
    完整流程:
    1. 模态对齐：将text和image投影到共同空间
    2. 跨模态交互：通过Transformer让两个模态相互增强
    3. 多种融合策略的组合
    
    这是老师说的"高质量"融合方法
    
    参数量说明：
    - lightweight (hidden=256): ~125万参数，适合CPU
    - balanced (hidden=384): ~280万参数，推荐用于小数据集
    - full (hidden=512): ~500万参数，适合大数据集
    """
    def __init__(
        self, 
        text_dim: int = 512,
        image_dim: int = 512, 
        common_dim: int = 512,
        num_transformer_layers: int = 2,  # 实际只用1层，避免过拟合
        nhead: int = 8,  # 实际用4头
        dropout: float = 0.1,
        hidden_dim: int = 384  # 新增：可配置内部维度（256/384/512）
    ):
        super().__init__()
        
        # 使用传入的hidden_dim，默认384（平衡版）
        self.common_dim = common_dim
        self.hidden_dim = hidden_dim
        
        # Step 1: 模态对齐层（轻量版）
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.image_projector = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Step 2: 轻量跨模态注意力（单层，4头）
        self.cross_attn_t2i = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_attn_i2t = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.norm_i = nn.LayerNorm(hidden_dim)
        
        # Step 3: 融合层（简化版）
        # 门控权重
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # 最终融合：gated + element-wise product
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, common_dim),
            nn.LayerNorm(common_dim)
        )
        
        self.output_dim = common_dim
        
    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor):
        """
        输入: 
            text_feat: [B, text_dim] 原始文本特征
            image_feat: [B, image_dim] 原始图像特征
        输出:
            fused: [B, common_dim] 融合后的多模态表征
        """
        # Step 1: 对齐到共同空间
        aligned_text = self.text_projector(text_feat)   # [B, hidden_dim]
        aligned_image = self.image_projector(image_feat)  # [B, hidden_dim]
        
        # Step 2: 跨模态注意力（轻量版）
        # 扩展维度: [B, hidden_dim] -> [B, 1, hidden_dim]
        text_seq = aligned_text.unsqueeze(1)
        image_seq = aligned_image.unsqueeze(1)
        
        # 文本关注图像
        text_attended, _ = self.cross_attn_t2i(text_seq, image_seq, image_seq)
        aligned_text = self.norm_t(aligned_text + text_attended.squeeze(1))
        
        # 图像关注文本
        image_attended, _ = self.cross_attn_i2t(image_seq, text_seq, text_seq)
        aligned_image = self.norm_i(aligned_image + image_attended.squeeze(1))
        
        # Step 3: 多策略融合
        # 3a. 门控融合
        concat_feat = torch.cat([aligned_text, aligned_image], dim=-1)
        gate_weights = self.gate(concat_feat)  # [B, 2]
        gated_fusion = gate_weights[:, 0:1] * aligned_text + gate_weights[:, 1:2] * aligned_image
        
        # 3b. 元素级交互
        element_fusion = aligned_text * aligned_image
        
        # 3c. 组合信号
        all_signals = torch.cat([gated_fusion, element_fusion, aligned_text + aligned_image], dim=-1)
        
        # 最终融合
        fused = self.final_fusion(all_signals)
        
        return fused


class ContrastiveAlignedFusion(nn.Module):
    """
    对比对齐融合 (Contrastive Aligned Fusion)
    
    在AlignedFusion基础上增加对比学习损失，
    进一步强制文本和图像在共同空间中对齐
    
    灵感来源：CLIP的对比学习思想
    """
    def __init__(
        self,
        text_dim: int = 512,
        image_dim: int = 512,
        common_dim: int = 512,
        num_transformer_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.aligned_fusion = AlignedFusion(
            text_dim, image_dim, common_dim,
            num_transformer_layers, nhead, dropout
        )
        
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        self.output_dim = common_dim
        
    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor):
        """正常前向传播"""
        return self.aligned_fusion(text_feat, image_feat)
    
    def forward_with_contrastive_loss(self, text_feat: torch.Tensor, image_feat: torch.Tensor):
        """
        带对比损失的前向传播
        
        返回:
            fused: 融合特征
            contrastive_loss: 对比损失（用于辅助训练）
        """
        # 获取对齐后的特征
        aligned_text, aligned_image = self.aligned_fusion.alignment(text_feat, image_feat)
        
        # L2归一化
        aligned_text_norm = F.normalize(aligned_text, dim=-1)
        aligned_image_norm = F.normalize(aligned_image, dim=-1)
        
        # 计算相似度矩阵
        logits = torch.matmul(aligned_text_norm, aligned_image_norm.T) / self.temperature
        
        # 对比损失：对角线元素应该最大（配对样本）
        batch_size = text_feat.shape[0]
        labels = torch.arange(batch_size, device=text_feat.device)
        
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.T, labels)
        contrastive_loss = (loss_t2i + loss_i2t) / 2
        
        # 正常的融合输出
        fused = self.aligned_fusion(text_feat, image_feat)
        
        return fused, contrastive_loss


class HierarchicalFusion(nn.Module):
    """
    层次化融合 (Hierarchical Fusion) - 轻量版
    
    多层次融合策略：
    - Level 1: 低层特征融合（浅层交互）
    - Level 2: 高层语义融合（深层理解）
    - Level 3: 决策层融合（最终整合）
    
    这种设计模拟人类理解多模态信息的层次性
    """
    def __init__(
        self,
        text_dim: int = 512,
        image_dim: int = 512,
        hidden_dim: int = 256,  # 会被忽略，使用更小的值
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 轻量化：使用更小的隐藏维度
        hidden_dim = 128  # 固定较小值
        
        # Level 1: 低层特征处理
        self.text_low = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.image_low = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Level 1 融合
        self.fusion_low = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Level 2: 高层语义处理（轻量注意力）
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        
        self.text_high = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.image_high = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Level 3: 决策层融合
        self.decision_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.output_dim = hidden_dim
        
    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor):
        # Level 1: 低层特征
        text_low = self.text_low(text_feat)
        image_low = self.image_low(image_feat)
        fusion_low = self.fusion_low(torch.cat([text_low, image_low], dim=-1))
        
        # Level 2: 高层语义 + 跨模态注意力
        text_high = self.text_high(text_low)
        image_high = self.image_high(image_low)
        
        # 跨模态注意力
        text_high_unsq = text_high.unsqueeze(1)
        image_high_unsq = image_high.unsqueeze(1)
        cross_out, _ = self.cross_attn(text_high_unsq, image_high_unsq, image_high_unsq)
        cross_out = cross_out.squeeze(1)
        
        # Level 3: 决策融合
        all_features = torch.cat([fusion_low, text_high, image_high, cross_out], dim=-1)
        output = self.decision_fusion(all_features)
        
        return output


# ========== 注册到融合方法字典 ==========
ADVANCED_FUSION_METHODS = {
    'aligned': AlignedFusion,
    'contrastive_aligned': ContrastiveAlignedFusion,
    'hierarchical': HierarchicalFusion,
}


def get_advanced_fusion(fusion_type: str, **kwargs):
    """获取高级融合模块"""
    if fusion_type not in ADVANCED_FUSION_METHODS:
        raise ValueError(f"Unknown fusion type: {fusion_type}. "
                        f"Available: {list(ADVANCED_FUSION_METHODS.keys())}")
    return ADVANCED_FUSION_METHODS[fusion_type](**kwargs)
