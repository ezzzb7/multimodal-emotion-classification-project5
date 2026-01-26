"""
正确的 Bad Case 分析 - 使用与训练时相同的数据加载方式
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import DistilBertTokenizer

sys.path.insert(0, os.path.dirname(__file__))

from run_experiment_optimized import OptimizedMultimodalClassifier, OPTIMIZED_CONFIG
from data.data_loader import get_data_loaders
from utils.train_utils import set_seed, compute_metrics


def analyze_bad_cases_correct(model_path):
    """使用与训练完全相同的数据加载方式"""
    
    print("\n" + "="*70)
    print("Bad Case 分析 (正确版)")
    print("="*70)
    
    config = OPTIMIZED_CONFIG.copy()
    config['dropout'] = 0.2  # HP1 配置
    
    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 使用与训练完全相同的数据加载
    print("加载数据（使用训练时相同的方式）...")
    train_loader, val_loader, _ = get_data_loaders(
        data_dir=config['data_dir'],
        train_label_file=config['train_label'],
        batch_size=1,  # batch_size=1 便于逐个分析
        val_ratio=config['val_ratio'],
        num_workers=0,
        seed=config['seed'],
        force_resplit=True
    )
    print(f"  验证集: {len(val_loader.dataset)} 样本")
    
    # 加载模型
    print(f"加载模型: {model_path}")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=True)
    
    model = OptimizedMultimodalClassifier(
        num_classes=3,
        feature_dim=config['feature_dim'],
        fusion_type='cross_attention',
        dropout=config['dropout'],
        unfreeze_text_layers=config['unfreeze_text_layers'],
        unfreeze_image_layers=config['unfreeze_image_layers']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    label_names = ['positive', 'negative', 'neutral']
    bad_cases = []
    all_preds = []
    all_labels = []
    
    # 混淆矩阵
    confusion = np.zeros((3, 3), dtype=int)
    
    print("开始分析...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='分析中'):
            guid = batch['guid'][0]
            text = batch['text'][0]
            image = batch['image'].to(device)
            label = batch['label'].item()
            
            encoded = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask, image)
            pred = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            all_preds.append(pred)
            all_labels.append(label)
            confusion[label][pred] += 1
            
            if pred != label:
                bad_cases.append({
                    'guid': guid,
                    'text': text[:200] + '...' if len(text) > 200 else text,
                    'true_label': label_names[label],
                    'pred_label': label_names[pred],
                    'confidence': probs[pred],
                    'prob_positive': probs[0],
                    'prob_negative': probs[1],
                    'prob_neutral': probs[2]
                })
    
    # 计算指标
    metrics = compute_metrics(all_preds, all_labels)
    
    print(f"\n验证集准确率: {metrics['accuracy']*100:.2f}%")
    print(f"验证集 F1: {metrics['f1']:.4f}")
    
    # 打印混淆矩阵
    print("\n混淆矩阵:")
    header = "真实\\预测"
    print(f"{header:<12} {'positive':<12} {'negative':<12} {'neutral':<12}")
    print("-"*48)
    for i, name in enumerate(label_names):
        row = f"{name:<12}"
        for j in range(3):
            row += f"{confusion[i][j]:<12}"
        print(row)
    
    # 错误分布
    print(f"\n错误样本数: {len(bad_cases)} / {len(all_labels)}")
    print("\n错误类型分布:")
    error_dist = {}
    for case in bad_cases:
        key = f"{case['true_label']} → {case['pred_label']}"
        error_dist[key] = error_dist.get(key, 0) + 1
    
    for k, v in sorted(error_dist.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} 个 ({v/len(bad_cases)*100:.1f}%)")
    
    # 打印部分 Bad Cases
    print(f"\nBad Cases 示例:")
    print("-"*80)
    for i, case in enumerate(bad_cases[:10]):
        print(f"\n[{i+1}] GUID: {case['guid']}")
        print(f"    文本: {case['text'][:60]}..." if len(case['text']) > 60 else f"    文本: {case['text']}")
        print(f"    真实: {case['true_label']} → 预测: {case['pred_label']} (置信度: {case['confidence']:.3f})")
    
    # 保存
    os.makedirs('analysis_results', exist_ok=True)
    df = pd.DataFrame(bad_cases)
    df.to_csv('analysis_results/bad_cases_hp1_best.csv', index=False, encoding='utf-8')
    print(f"\n✓ Bad cases 已保存到: analysis_results/bad_cases_hp1_best.csv")
    
    return metrics, bad_cases


def generate_test_predictions(model_path, output_file='predictions.txt'):
    """生成测试集预测"""
    
    print("\n" + "="*70)
    print("生成测试集预测")
    print("="*70)
    
    config = OPTIMIZED_CONFIG.copy()
    config['dropout'] = 0.2
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=True)
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = OptimizedMultimodalClassifier(
        num_classes=3,
        feature_dim=config['feature_dim'],
        fusion_type='cross_attention',
        dropout=config['dropout'],
        unfreeze_text_layers=config['unfreeze_text_layers'],
        unfreeze_image_layers=config['unfreeze_image_layers']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 加载测试集
    from data.dataset import MultimodalDataset
    from data.preprocessing import TextPreprocessor, ImagePreprocessor
    
    test_file = r'D:\当代人工智能\project5\test_without_label.txt'
    
    text_preprocessor = TextPreprocessor()
    image_preprocessor = ImagePreprocessor(mode='val')
    
    test_dataset = MultimodalDataset(
        data_dir=config['data_dir'],
        label_file=test_file,
        text_preprocessor=text_preprocessor,
        image_preprocessor=image_preprocessor,
        mode='test'
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=0
    )
    
    print(f"测试集样本数: {len(test_dataset)}")
    
    label_names = ['positive', 'negative', 'neutral']
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='预测中'):
            guids = batch['guid']
            texts = batch['text']
            images = batch['image'].to(device)
            
            encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask, images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            for guid, pred in zip(guids, preds):
                predictions.append((guid, label_names[pred]))
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('guid,tag\n')
        for guid, tag in predictions:
            f.write(f'{guid},{tag}\n')
    
    print(f"\n✓ 预测结果已保存到: {output_file}")
    
    # 统计
    pred_dist = {}
    for _, tag in predictions:
        pred_dist[tag] = pred_dist.get(tag, 0) + 1
    
    print("\n预测分布:")
    for tag, count in sorted(pred_dist.items()):
        print(f"  {tag}: {count} ({count/len(predictions)*100:.1f}%)")
    
    return predictions


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--badcase', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--model', type=str, default='experiments/checkpoints/HP1_BEST_best.pth')
    parser.add_argument('--output', type=str, default='predictions.txt')
    
    args = parser.parse_args()
    
    if args.all:
        analyze_bad_cases_correct(args.model)
        generate_test_predictions(args.model, args.output)
    elif args.badcase:
        analyze_bad_cases_correct(args.model)
    elif args.predict:
        generate_test_predictions(args.model, args.output)
    else:
        print("使用方法:")
        print("  py -3.11 run_correct_analysis.py --badcase")
        print("  py -3.11 run_correct_analysis.py --predict")
        print("  py -3.11 run_correct_analysis.py --all")
