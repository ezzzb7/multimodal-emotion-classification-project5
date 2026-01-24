"""
Bad Case 分析工具
找出模型预测错误的样本，分析共同特征，为针对性优化提供依据

⚠️ 重要：只分析训练集，避免验证集信息泄露
   - 验证集用于评估模型性能，不能参与训练数据的选择
   - 数据增强只能基于训练集的分析结果
"""
import torch
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from models.multimodal_model import MultimodalClassifier
from data.data_loader import get_data_loaders
from utils.train_utils import load_checkpoint


def analyze_bad_cases(model, dataloader, device='cpu', label_map=None):
    """
    分析模型的错误预测
    
    Returns:
        bad_cases: list of dicts with error info
        stats: dict with statistics
    """
    model.eval()
    bad_cases = []
    
    label_names = {0: 'positive', 1: 'negative', 2: 'neutral'}
    if label_map:
        label_names = {v: k for k, v in label_map.items()}
    
    with torch.no_grad():
        for batch in dataloader:
            texts = batch['text']
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            guids = batch['guid']
            
            outputs = model(texts, images)
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)
            
            # 找出错误预测（只保留高置信度错误，避免噪声样本）
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    # 确保GUID是字符串或整数，而不是tensor
                    guid = guids[i]
                    if isinstance(guid, torch.Tensor):
                        guid = guid.item()
                    
                    # 计算预测置信度
                    pred_confidence = probs[i][preds[i]].item()
                    
                    # 只保留高置信度错误（>0.7），这些更可能是真正的困难样本
                    if pred_confidence > 0.7:
                        bad_case = {
                            'guid': guid,
                            'text': texts[i][:200],  # 截取前200字符
                            'true_label': label_names[labels[i].item()],
                            'pred_label': label_names[preds[i].item()],
                            'confidence': pred_confidence,
                            'true_prob': probs[i][labels[i]].item(),
                            'probs': probs[i].cpu().numpy()
                        }
                        bad_cases.append(bad_case)
    
    # 统计分析
    stats = analyze_error_patterns(bad_cases)
    
    return bad_cases, stats


def analyze_error_patterns(bad_cases):
    """分析错误模式"""
    stats = {
        'total_errors': len(bad_cases),
        'confusion_matrix': defaultdict(int),
        'low_confidence_errors': 0,
        'high_confidence_errors': 0,
        'avg_confidence': 0,
        'text_length_stats': []
    }
    
    if not bad_cases:
        return stats
    
    confidences = []
    text_lengths = []
    
    for case in bad_cases:
        # 混淆矩阵
        key = f"{case['true_label']} → {case['pred_label']}"
        stats['confusion_matrix'][key] += 1
        
        # 置信度统计
        conf = case['confidence']
        confidences.append(conf)
        
        if conf < 0.5:
            stats['low_confidence_errors'] += 1
        else:
            stats['high_confidence_errors'] += 1
        
        # 文本长度
        text_lengths.append(len(case['text']))
    
    stats['avg_confidence'] = np.mean(confidences)
    stats['text_length_stats'] = {
        'mean': np.mean(text_lengths),
        'std': np.std(text_lengths),
        'min': np.min(text_lengths),
        'max': np.max(text_lengths)
    }
    
    return stats


def save_bad_cases(bad_cases, output_path='bad_cases_analysis.csv'):
    """保存bad cases到CSV"""
    if not bad_cases:
        print("No bad cases found!")
        return
    
    df = pd.DataFrame(bad_cases)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✓ Bad cases saved to: {output_path}")


def print_analysis_report(stats, bad_cases):
    """打印分析报告"""
    print("\n" + "="*70)
    print("Bad Case Analysis Report")
    print("="*70)
    
    print(f"\n📊 总体统计:")
    print(f"  - 错误样本数: {stats['total_errors']}")
    print(f"  - 平均预测置信度: {stats['avg_confidence']:.4f}")
    print(f"  - 低置信度错误 (<0.5): {stats['low_confidence_errors']}")
    print(f"  - 高置信度错误 (≥0.5): {stats['high_confidence_errors']}")
    
    print(f"\n🔀 混淆矩阵 (错误类型分布):")
    for error_type, count in sorted(stats['confusion_matrix'].items(), key=lambda x: -x[1])[:10]:
        percentage = count / stats['total_errors'] * 100
        print(f"  {error_type}: {count} ({percentage:.1f}%)")
    
    print(f"\n📏 文本长度统计:")
    tl = stats['text_length_stats']
    print(f"  - 平均长度: {tl['mean']:.1f} ± {tl['std']:.1f}")
    print(f"  - 范围: [{tl['min']}, {tl['max']}]")
    
    # 高置信度错误案例（最值得关注）
    print(f"\n⚠️ 高置信度错误案例 (Top 5):")
    high_conf_errors = sorted([bc for bc in bad_cases if bc['confidence'] >= 0.5],
                             key=lambda x: -x['confidence'])[:5]
    for i, case in enumerate(high_conf_errors, 1):
        print(f"\n  {i}. GUID: {case['guid']}")
        print(f"     真实: {case['true_label']} | 预测: {case['pred_label']} (置信度: {case['confidence']:.3f})")
        print(f"     文本: {case['text'][:100]}...")
    
    print("\n" + "="*70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Bad Case分析工具')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                        help='分析哪个数据集 (默认: train, 推荐只用train避免信息泄露)')
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/best_early_multimodal_20260120_195503.pth',
                        help='模型checkpoint路径')
    args = parser.parse_args()
    
    # 配置
    checkpoint_path = args.checkpoint
    data_dir = r'D:\当代人工智能\project5\data'
    train_file = r'D:\当代人工智能\project5\train.txt'
    device = 'cpu'
    
    print("="*70)
    print("Bad Case Analysis Tool")
    print("="*70)
    print(f"\n⚠️ 分析数据集: {args.split.upper()}")
    
    if args.split == 'val':
        print("   警告: 分析验证集可能导致信息泄露，建议使用 --split train")
    else:
        print("   ✓ 正确: 只分析训练集，避免验证集信息泄露")
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, _ = get_data_loaders(
        data_dir=data_dir,
        train_label_file=train_file,
        batch_size=8,
        val_ratio=0.15,
        num_workers=0,
        seed=42
    )
    
    # 选择数据集
    target_loader = train_loader if args.split == 'train' else val_loader
    
    # 创建模型
    print("创建模型...")
    model = MultimodalClassifier(
        num_classes=3,
        text_model='distilbert-base-uncased',
        image_model='resnet50',
        fusion_type='early',
        feature_dim=512,
        freeze_encoders=True,
        dropout=0.3
    ).to(device)
    
    # 加载checkpoint
    if os.path.exists(checkpoint_path):
        print(f"加载checkpoint: {checkpoint_path}")
        model, _, _, _ = load_checkpoint(model, None, checkpoint_path)
    else:
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        print("使用未训练的模型进行分析（仅用于测试）")
    
    # 分析bad cases
    print(f"\n分析{args.split}集错误样本...")
    bad_cases, stats = analyze_bad_cases(model, target_loader, device)
    
    # 保存结果
    output_path = f'analysis_results/bad_cases_{args.split}.csv'
    save_bad_cases(bad_cases, output_path)
    
    # 打印报告
    print_analysis_report(stats, bad_cases)
    
    # 生成优化建议
    print("\n💡 优化建议:")
    
    if args.split == 'train':
        print("  ✓ 可以安全地使用这些bad cases进行数据增强")
        print("  ✓ 运行: python augment_bad_cases.py --input analysis_results/bad_cases_train.csv")
    else:
        print("  ⚠️ 这些是验证集的bad cases，仅供分析参考")
        print("  ⚠️ 不要使用这些数据进行训练或增强！")
    
    if stats['high_confidence_errors'] > stats['total_errors'] * 0.3:
        print("\n  ⚠️ 高置信度错误较多 → 建议:")
        print("     - 检查数据标注质量")
        print("     - 对高置信度错误样本做数据增强")
        print("     - 考虑增加模型容量")
    
    confusion = stats['confusion_matrix']
    if confusion.get('neutral → positive', 0) + confusion.get('neutral → negative', 0) > stats['total_errors'] * 0.3:
        print("\n  ⚠️ neutral类别容易误判 → 建议:")
        print("     - 增加neutral样本的数据增强")
        print("     - 调整类别权重")
        print("     - 使用focal loss")


if __name__ == '__main__':
    os.makedirs('analysis_results', exist_ok=True)
    main()
