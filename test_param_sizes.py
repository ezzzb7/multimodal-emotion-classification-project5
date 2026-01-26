"""测试不同配置的参数量"""
import torch
from models.advanced_fusion import AlignedFusion, HierarchicalFusion

print('='*60)
print('AlignedFusion 参数量对比')
print('='*60)

for h in [256, 384, 512]:
    af = AlignedFusion(512, 512, 512, hidden_dim=h)
    params = sum(p.numel() for p in af.parameters())
    print(f'hidden_dim={h:3d}: {params:>10,} 参数 ({params/1e6:.2f}M)')

print()
print('='*60)
print('对比基础融合方法')
print('='*60)
print('Late Fusion:        ~1,705,987 参数 (1.7M) → 67.00% 准确率')
print('Cross-Attention:    ~2,428,675 参数 (2.4M) → 66.75% 准确率')
print('Early Fusion:       ~2,625,027 参数 (2.6M) → 64.12% 准确率')

print()
print('='*60)
print('推荐配置')
print('='*60)
print('✓ hidden_dim=384 (平衡版): ~2.8M 参数')
print('  优势:')
print('  - 比Late Fusion多60%参数，但设计更合理（解决空间对齐问题）')
print('  - 在3200样本训练集上不会过拟合')
print('  - 包含模态对齐层 + 跨模态注意力 + 多策略融合')
print('  - 预期准确率: 68-70% (提升1-3%)')
print()
print('✗ hidden_dim=512 (完整版): ~5M 参数')
print('  问题: 参数过多，在小数据集上容易过拟合')
print()
print('✓ hidden_dim=256 (轻量版): ~1.3M 参数')
print('  适用: CPU训练，但可能欠拟合')
