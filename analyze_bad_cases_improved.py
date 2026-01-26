"""
改进的Bad Case分析工具 - 使用训练集避免数据泄漏
推荐实践：使用训练集高置信度错误进行增强
"""

import os
import sys
import argparse
import torch
import pandas as pd
from collections import Counter
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.multimodal_model import MultimodalClassifier
from data.data_loader import get_data_loaders


def analyze_bad_cases_improved(checkpoint_path, data_dir, output_dir='analysis_results',
                               split='train', min_confidence=0.7):
    """
    改进的Bad Case分析
    
    Args:
        checkpoint_path: 模型checkpoint路径
        data_dir: 数据目录
        output_dir: 输出目录
        split: 'train' (推荐，避免数据泄漏) 或 'val'
        min_confidence: 只分析预测置信度>此值的错误样本
    
    推荐配置:
        - split='train': 使用训练集bad cases，避免验证集信息泄漏
        - min_confidence=0.7: 只增强高置信度错误（模型确信但错了的样本）
    """
    print("=" * 70)
    print("改进的Bad Case分析工具")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  - 数据集: {split} (推荐使用train避免数据泄漏)")
    print(f"  - 最小置信度: {min_confidence} (只分析高置信度错误)")
    print(f"  - Checkpoint: {checkpoint_path}\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    train_label_file = os.path.join(data_dir, 'train.txt')
    train_loader, val_loader, _ = get_data_loaders(
        data_dir=data_dir,
        train_label_file=train_label_file,
        batch_size=8
    )
    dataloader = train_loader if split == 'train' else val_loader
    
    # 加载模型
    print("\n创建模型...")
    model = MultimodalClassifier(
        num_classes=3,
        fusion_type='early',
        freeze_encoders=True
    )
    
    print(f"\n加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Checkpoint loaded: epoch {checkpoint['epoch']}, best_acc {checkpoint['best_acc']:.4f}")
    
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    
    # 分析错误
    print(f"\n分析{split}集错误（最小置信度: {min_confidence}）...")
    bad_cases = []
    label_names = ['positive', 'negative', 'neutral']
    
    with torch.no_grad():
        for batch in dataloader:
            # 前向传播
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # 找出错误样本
            errors = preds != batch['label']
            
            for i in range(len(batch['label'])):
                if errors[i]:
                    confidence = probs[i, preds[i]].item()
                    
                    # 只保存高置信度错误
                    if confidence >= min_confidence:
                        guid = batch['guid'][i]
                        if isinstance(guid, torch.Tensor):
                            guid = guid.item()
                        
                        bad_cases.append({
                            'guid': guid,
                            'text': batch['text'][i],
                            'true_label': label_names[batch['label'][i].item()],
                            'pred_label': label_names[preds[i].item()],
                            'confidence': confidence
                        })
    
    print(f"✓ 找到 {len(bad_cases)} 个高置信度错误样本（置信度 > {min_confidence}）")
    
    if len(bad_cases) == 0:
        print("⚠️ 没有找到满足条件的bad cases")
        return
    
    # 保存bad cases
    output_path = os.path.join(output_dir, 'bad_cases.csv')
    df = pd.DataFrame(bad_cases)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✓ Bad cases saved to: {output_path}")
    
    # 统计分析
    print("\n" + "=" * 70)
    print("Bad Case分析报告")
    print("=" * 70)
    
    print(f"\n📊 总体统计:")
    print(f"  - 数据集: {split}")
    print(f"  - 错误样本数: {len(bad_cases)}")
    print(f"  - 平均预测置信度: {np.mean([c['confidence'] for c in bad_cases]):.4f}")
    
    # 混淆分析
    print(f"\n🔀 混淆矩阵 (错误类型分布):")
    confusion = Counter([(c['true_label'], c['pred_label']) for c in bad_cases])
    for (true_label, pred_label), count in confusion.most_common():
        pct = count / len(bad_cases) * 100
        print(f"  {true_label} → {pred_label}: {count} ({pct:.1f}%)")
    
    # 文本长度分析
    text_lengths = [len(c['text']) for c in bad_cases]
    print(f"\n📏 文本长度统计:")
    print(f"  - 平均长度: {np.mean(text_lengths):.1f} ± {np.std(text_lengths):.1f}")
    print(f"  - 范围: [{min(text_lengths)}, {max(text_lengths)}]")
    
    # 高置信度错误案例
    print(f"\n⚠️ 高置信度错误案例 (Top 5):")
    sorted_cases = sorted(bad_cases, key=lambda x: x['confidence'], reverse=True)
    for i, case in enumerate(sorted_cases[:5], 1):
        print(f"\n  {i}. GUID: {case['guid']}")
        print(f"     真实: {case['true_label']} | 预测: {case['pred_label']} (置信度: {case['confidence']:.3f})")
        print(f"     文本: {case['text'][:100]}...")
    
    print("\n" + "=" * 70)
    
    print(f"\n💡 数据泄漏风险评估:")
    if split == 'val':
        print("  ⚠️ 警告: 使用验证集会导致信息泄漏")
        print("  ✓ 建议: 使用训练集bad cases进行增强")
    else:
        print("  ✓ 良好实践: 使用训练集避免了验证集泄漏")
    
    print(f"\n💡 优化建议:")
    print(f"  ✓ 找到 {len(bad_cases)} 个高置信度错误（模型确信但错了）")
    print(f"  ✓ 建议对这些样本进行适度增强（2-3倍）")
    print(f"  ✓ 避免过度增强导致过拟合")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='改进的Bad Case分析')
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints/best_early_multimodal_20260120_195503.pth',
                       help='模型checkpoint路径')
    parser.add_argument('--data_dir', type=str,
                       default=r'D:\当代人工智能\project5\data',
                       help='数据目录')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val'],
                       help='使用哪个数据集（推荐train避免数据泄漏）')
    parser.add_argument('--min_confidence', type=float, default=0.7,
                       help='最小预测置信度（只分析高置信度错误）')
    
    args = parser.parse_args()
    
    analyze_bad_cases_improved(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir='analysis_results',
        split=args.split,
        min_confidence=args.min_confidence
    )


if __name__ == '__main__':
    main()
