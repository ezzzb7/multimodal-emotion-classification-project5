"""
完整实验流程脚本
包含：数据增强对比、Bad Case分析、测试集预测
"""
import os
import sys
import time
import csv
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, DistilBertTokenizer
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(__file__))

from utils.train_utils import set_seed, compute_metrics, EarlyStopping
from run_experiment_optimized import OptimizedMultimodalClassifier, OPTIMIZED_CONFIG


# ========== 数据增强配置 ==========
# 基础transform（无增强）
BASIC_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 增强transform
AUGMENTED_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证/测试用的transform（无增强）
VAL_TRANSFORM = BASIC_TRANSFORM


def clean_text(text: str) -> str:
    """
    文本清洗
    - 移除多余空格
    - 移除特殊字符
    - 统一大小写
    """
    import re
    
    if not isinstance(text, str):
        return ""
    
    # 移除URL
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # 移除@mentions
    text = re.sub(r'@\w+', '', text)
    
    # 移除#hashtags（保留文字）
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空格
    text = text.strip()
    
    return text


class AugmentedDataset(torch.utils.data.Dataset):
    """支持数据增强的数据集"""
    
    def __init__(self, data_dir, split_file, transform=None, text_clean=False):
        self.data_dir = data_dir
        self.transform = transform or BASIC_TRANSFORM
        self.text_clean = text_clean
        
        self.samples = []
        self.label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    guid = parts[0]
                    label = parts[1]
                    self.samples.append((guid, self.label_map.get(label, 0)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        guid, label = self.samples[idx]
        
        # 加载文本
        text_path = os.path.join(self.data_dir, f"{guid}.txt")
        try:
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
        except:
            text = ""
        
        # 文本清洗
        if self.text_clean:
            text = clean_text(text)
        
        # 加载图像
        image_path = os.path.join(self.data_dir, f"{guid}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except:
            image = torch.zeros(3, 224, 224)
        
        return {
            'guid': guid,
            'text': text,
            'image': image,
            'label': label
        }


class TestDataset(torch.utils.data.Dataset):
    """测试集数据集（无标签）"""
    
    def __init__(self, data_dir, test_file, transform=None, text_clean=False):
        self.data_dir = data_dir
        self.transform = transform or VAL_TRANSFORM
        self.text_clean = text_clean
        
        self.samples = []
        with open(test_file, 'r', encoding='utf-8') as f:
            next(f)  # 跳过header
            for line in f:
                guid = line.strip().split(',')[0]
                if guid:
                    self.samples.append(guid)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        guid = self.samples[idx]
        
        # 加载文本
        text_path = os.path.join(self.data_dir, f"{guid}.txt")
        try:
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
        except:
            text = ""
        
        if self.text_clean:
            text = clean_text(text)
        
        # 加载图像
        image_path = os.path.join(self.data_dir, f"{guid}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except:
            image = torch.zeros(3, 224, 224)
        
        return {
            'guid': guid,
            'text': text,
            'image': image
        }


def run_augmentation_experiment():
    """
    数据增强对比实验
    对比：无增强 vs 图像增强 vs 文本清洗 vs 全部增强
    """
    print("\n" + "="*70)
    print("数据增强对比实验")
    print("="*70)
    
    config = OPTIMIZED_CONFIG
    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    experiments = [
        ('AUG_none', '无增强', BASIC_TRANSFORM, False),
        ('AUG_image', '图像增强', AUGMENTED_TRANSFORM, False),
        ('AUG_text', '文本清洗', BASIC_TRANSFORM, True),
        ('AUG_both', '全部增强', AUGMENTED_TRANSFORM, True),
    ]
    
    results = []
    
    for exp_id, exp_name, img_transform, text_clean in experiments:
        print(f"\n{'='*50}")
        print(f"实验 {exp_id}: {exp_name}")
        print(f"{'='*50}")
        
        # 创建数据集
        train_dataset = AugmentedDataset(
            data_dir=config['data_dir'],
            split_file='splits/train_split.txt',
            transform=img_transform,
            text_clean=text_clean
        )
        val_dataset = AugmentedDataset(
            data_dir=config['data_dir'],
            split_file='splits/val_split.txt',
            transform=VAL_TRANSFORM,  # 验证集不增强
            text_clean=text_clean
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0
        )
        
        print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
        
        # 创建模型
        model = OptimizedMultimodalClassifier(
            num_classes=3,
            feature_dim=config['feature_dim'],
            fusion_type='late',
            dropout=config['dropout'],
            unfreeze_text_layers=config['unfreeze_text_layers'],
            unfreeze_image_layers=config['unfreeze_image_layers']
        ).to(device)
        
        # 训练设置
        class_weights = torch.FloatTensor([1.0, 1.5, 3.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        param_groups = model.get_param_groups(
            lr_pretrained=config['lr_pretrained'],
            lr_fusion=config['lr_fusion'],
            lr_classifier=config['lr_classifier']
        )
        optimizer = AdamW(param_groups, weight_decay=config['weight_decay'])
        
        total_steps = len(train_loader) * config['num_epochs'] // config['accumulation_steps']
        warmup_steps = int(total_steps * config['warmup_ratio'])
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        
        early_stopping = EarlyStopping(patience=config['early_stopping_patience'], mode='max')
        
        # 训练
        best_val_acc = 0
        best_epoch = 0
        start_time = time.time()
        
        for epoch in range(config['num_epochs']):
            # 训练
            model.train()
            train_preds, train_labels = [], []
            
            for step, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)):
                texts = batch['text']
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                logits = model(input_ids, attention_mask, images)
                loss = criterion(logits, labels) / config['accumulation_steps']
                loss.backward()
                
                if (step + 1) % config['accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
            
            # 验证
            model.eval()
            val_preds, val_labels = [], []
            
            with torch.no_grad():
                for batch in val_loader:
                    texts = batch['text']
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)
                    
                    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
                    input_ids = encoded['input_ids'].to(device)
                    attention_mask = encoded['attention_mask'].to(device)
                    
                    logits = model(input_ids, attention_mask, images)
                    val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            val_metrics = compute_metrics(val_preds, val_labels)
            print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_metrics['accuracy']:.4f}")
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_val_f1 = val_metrics['f1']
                best_epoch = epoch + 1
                torch.save(model.state_dict(), f'experiments/checkpoints/{exp_id}_best.pth')
                print(f"  ✓ 新最佳!")
            
            if early_stopping(val_metrics['accuracy'], epoch):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        training_time = (time.time() - start_time) / 60
        
        results.append({
            'exp_id': exp_id,
            'exp_name': exp_name,
            'val_acc': best_val_acc,
            'val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'training_time': training_time
        })
        
        print(f"实验 {exp_id} 完成! Val Acc: {best_val_acc:.4f}")
    
    # 打印对比结果
    print("\n" + "="*70)
    print("数据增强对比结果")
    print("="*70)
    print(f"{'实验':<15} {'描述':<15} {'Val Acc':<10} {'Val F1':<10}")
    print("-"*50)
    for r in results:
        print(f"{r['exp_id']:<15} {r['exp_name']:<15} {r['val_acc']:.4f}     {r['val_f1']:.4f}")
    
    return results


def analyze_bad_cases(model_path: str, num_cases: int = 50):
    """
    分析Bad Cases
    找出模型预测错误的样本，分析错误模式
    """
    print("\n" + "="*70)
    print("Bad Case 分析")
    print("="*70)
    
    config = OPTIMIZED_CONFIG
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 加载最佳模型
    model = OptimizedMultimodalClassifier(
        num_classes=3,
        feature_dim=config['feature_dim'],
        fusion_type='late',
        dropout=config['dropout'],
        unfreeze_text_layers=config['unfreeze_text_layers'],
        unfreeze_image_layers=config['unfreeze_image_layers']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 使用训练集分析（避免信息泄露）
    train_dataset = AugmentedDataset(
        data_dir=config['data_dir'],
        split_file='splits/train_split.txt',
        transform=VAL_TRANSFORM,
        text_clean=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    
    label_names = ['positive', 'negative', 'neutral']
    bad_cases = []
    confusion = {'positive': {}, 'negative': {}, 'neutral': {}}
    
    with torch.no_grad():
        for batch in tqdm(train_loader, desc='分析中'):
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
            
            true_label = label_names[label]
            pred_label = label_names[pred]
            
            # 记录混淆
            if pred_label not in confusion[true_label]:
                confusion[true_label][pred_label] = 0
            confusion[true_label][pred_label] += 1
            
            # 记录错误样本
            if pred != label:
                bad_cases.append({
                    'guid': guid,
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': probs[pred],
                    'probs': f"pos={probs[0]:.2f}, neg={probs[1]:.2f}, neu={probs[2]:.2f}"
                })
    
    # 打印混淆矩阵
    print("\n混淆矩阵 (训练集):")
    print(f"{'True \\ Pred':<12} {'positive':<12} {'negative':<12} {'neutral':<12}")
    print("-"*48)
    for true_label in label_names:
        row = f"{true_label:<12}"
        for pred_label in label_names:
            count = confusion[true_label].get(pred_label, 0)
            row += f"{count:<12}"
        print(row)
    
    # 打印错误样本
    print(f"\n错误样本示例 (共 {len(bad_cases)} 个):")
    print("-"*80)
    
    for i, case in enumerate(bad_cases[:num_cases]):
        print(f"\n[{i+1}] GUID: {case['guid']}")
        print(f"    文本: {case['text']}")
        print(f"    真实: {case['true_label']} → 预测: {case['pred_label']} (置信度: {case['confidence']:.2f})")
        print(f"    概率: {case['probs']}")
    
    # 保存到CSV
    os.makedirs('analysis_results', exist_ok=True)
    df = pd.DataFrame(bad_cases)
    df.to_csv('analysis_results/bad_cases_analysis.csv', index=False, encoding='utf-8')
    print(f"\n✓ Bad cases 已保存到: analysis_results/bad_cases_analysis.csv")
    
    # 错误分布统计
    print("\n错误分布统计:")
    error_dist = {}
    for case in bad_cases:
        key = f"{case['true_label']} → {case['pred_label']}"
        error_dist[key] = error_dist.get(key, 0) + 1
    
    for k, v in sorted(error_dist.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} 个 ({v/len(bad_cases)*100:.1f}%)")
    
    return bad_cases


def generate_test_predictions(model_path: str, output_file: str = 'predictions.txt'):
    """
    生成测试集预测结果
    """
    print("\n" + "="*70)
    print("生成测试集预测")
    print("="*70)
    
    config = OPTIMIZED_CONFIG
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 加载最佳模型
    print(f"加载模型: {model_path}")
    model = OptimizedMultimodalClassifier(
        num_classes=3,
        feature_dim=config['feature_dim'],
        fusion_type='late',  # 使用最佳配置
        dropout=config['dropout'],
        unfreeze_text_layers=config['unfreeze_text_layers'],
        unfreeze_image_layers=config['unfreeze_image_layers']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 加载测试集
    test_file = r'D:\当代人工智能\project5\test_without_label.txt'
    test_dataset = TestDataset(
        data_dir=config['data_dir'],
        test_file=test_file,
        transform=VAL_TRANSFORM,
        text_clean=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0
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
    
    # 保存预测结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('guid,tag\n')
        for guid, tag in predictions:
            f.write(f'{guid},{tag}\n')
    
    print(f"\n✓ 预测结果已保存到: {output_file}")
    
    # 统计预测分布
    pred_dist = {}
    for _, tag in predictions:
        pred_dist[tag] = pred_dist.get(tag, 0) + 1
    
    print("\n预测分布:")
    for tag, count in sorted(pred_dist.items()):
        print(f"  {tag}: {count} ({count/len(predictions)*100:.1f}%)")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='完整实验流程')
    parser.add_argument('--augmentation', action='store_true', help='运行数据增强对比实验')
    parser.add_argument('--badcase', action='store_true', help='分析Bad Cases')
    parser.add_argument('--predict', action='store_true', help='生成测试集预测')
    parser.add_argument('--model', type=str, default='experiments/checkpoints/OPT_late_optimized_late_best.pth',
                       help='模型路径')
    parser.add_argument('--output', type=str, default='predictions.txt', help='预测输出文件')
    
    args = parser.parse_args()
    
    if args.augmentation:
        run_augmentation_experiment()
    elif args.badcase:
        analyze_bad_cases(args.model)
    elif args.predict:
        generate_test_predictions(args.model, args.output)
    else:
        print("使用方法:")
        print("  py -3.11 run_complete_experiments.py --augmentation  # 数据增强对比")
        print("  py -3.11 run_complete_experiments.py --badcase       # Bad Case分析")
        print("  py -3.11 run_complete_experiments.py --predict       # 生成测试集预测")


if __name__ == '__main__':
    main()
