"""
完整流水线：重新训练最佳配置 + Bad Case分析 + 测试集预测
整合所有剩余任务到一个脚本中
"""
import os
import sys
import re
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, DistilBertTokenizer
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from run_experiment_optimized import OptimizedMultimodalClassifier, OPTIMIZED_CONFIG
from utils.train_utils import set_seed, compute_metrics, EarlyStopping


# ========== 配置 ==========
BEST_CONFIG = {
    **OPTIMIZED_CONFIG,
    'dropout': 0.2,  # HP1最佳配置
    'lr_classifier': 1e-4,
    'weight_decay': 0.01,
}


# ========== 文本清洗 ==========
def clean_text(text):
    """文本清洗"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ========== 图像Transform ==========
BASIC_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ========== 数据集 ==========
class MultimodalDataset(Dataset):
    def __init__(self, data_dir, split_file, transform=None, has_label=True):
        self.data_dir = data_dir
        self.transform = transform or BASIC_TRANSFORM
        self.has_label = has_label
        
        self.samples = []
        self.label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 1:
                    guid = parts[0]
                    if guid == 'guid':  # 跳过header
                        continue
                    label = self.label_map.get(parts[1], 0) if has_label and len(parts) >= 2 else None
                    self.samples.append((guid, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        guid, label = self.samples[idx]
        
        # 文本
        text_path = os.path.join(self.data_dir, f"{guid}.txt")
        try:
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = clean_text(f.read().strip())
        except:
            text = ""
        
        # 图像
        image_path = os.path.join(self.data_dir, f"{guid}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except:
            image = torch.zeros(3, 224, 224)
        
        result = {'guid': guid, 'text': text, 'image': image}
        if self.has_label:
            result['label'] = label
        return result


# ========== 1. 训练最佳模型并保存 ==========
def train_best_model():
    """使用最佳配置训练模型并保存"""
    
    print("\n" + "="*70)
    print("训练最佳配置模型 (HP1: Dropout=0.2)")
    print("="*70)
    
    config = BEST_CONFIG
    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=True)
    
    # 数据
    train_dataset = MultimodalDataset(config['data_dir'], 'splits/train_split.txt')
    val_dataset = MultimodalDataset(config['data_dir'], 'splits/val_split.txt')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    # 模型
    model = OptimizedMultimodalClassifier(
        num_classes=3,
        feature_dim=config['feature_dim'],
        fusion_type='cross_attention',
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
    
    total_steps = len(train_loader) * 20 // config['accumulation_steps']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    early_stopping = EarlyStopping(patience=8, mode='max')
    
    # 训练
    print("\n开始训练...")
    sys.stdout.flush()
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(20):
        model.train()
        print(f"Epoch {epoch+1}/20 [", end="", flush=True)
        
        for step, batch in enumerate(train_loader):
            texts = batch['text']
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors='pt')
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
            
            if (step + 1) % 20 == 0:
                print(".", end="", flush=True)
        
        print("] ", end="", flush=True)
        
        # 验证
        model.eval()
        val_preds, val_labels_list = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                texts = batch['text']
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors='pt')
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                logits = model(input_ids, attention_mask, images)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_metrics = compute_metrics(val_preds, val_labels_list)
        print(f"Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}", flush=True)
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
            print(f"  ✓ 新最佳!", flush=True)
        
        if early_stopping(val_metrics['accuracy'], epoch):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 保存最佳模型
    os.makedirs('experiments/checkpoints', exist_ok=True)
    model_path = 'experiments/checkpoints/BEST_HP1_cross_attention.pth'
    torch.save(best_model_state, model_path)
    print(f"\n✓ 最佳模型已保存: {model_path}")
    print(f"  最佳 Val Acc: {best_val_acc:.4f}")
    
    return model_path, best_val_acc


# ========== 2. Bad Case分析 ==========
def analyze_bad_cases(model_path):
    """分析验证集上的Bad Cases"""
    
    print("\n" + "="*70)
    print("Bad Case 分析")
    print("="*70)
    
    config = BEST_CONFIG
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
    
    # 加载验证集
    val_dataset = MultimodalDataset(config['data_dir'], 'splits/val_split.txt')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    label_names = ['positive', 'negative', 'neutral']
    bad_cases = []
    all_predictions = []
    
    # 混淆矩阵
    confusion = np.zeros((3, 3), dtype=int)
    
    print(f"分析 {len(val_dataset)} 个验证样本...")
    
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
            
            confusion[label][pred] += 1
            
            all_predictions.append({
                'guid': guid,
                'true': label_names[label],
                'pred': label_names[pred],
                'correct': pred == label,
                'confidence': probs[pred]
            })
            
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
    
    # 打印结果
    total = len(val_dataset)
    correct = sum(1 for p in all_predictions if p['correct'])
    print(f"\n验证集准确率: {correct}/{total} ({correct/total*100:.2f}%)")
    
    print("\n混淆矩阵:")
    header = "真实\\预测"
    print(f"{header:<12} {'positive':<12} {'negative':<12} {'neutral':<12}")
    print("-"*48)
    for i, name in enumerate(label_names):
        row = f"{name:<12}"
        for j in range(3):
            row += f"{confusion[i][j]:<12}"
        print(row)
    
    # 错误类型分布
    print("\n错误类型分布:")
    error_dist = {}
    for case in bad_cases:
        key = f"{case['true_label']} → {case['pred_label']}"
        error_dist[key] = error_dist.get(key, 0) + 1
    
    for k, v in sorted(error_dist.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} 个 ({v/len(bad_cases)*100:.1f}%)")
    
    # 打印部分Bad Cases
    print(f"\nBad Cases 示例 (共 {len(bad_cases)} 个):")
    print("-"*80)
    for i, case in enumerate(bad_cases[:15]):
        print(f"\n[{i+1}] GUID: {case['guid']}")
        print(f"    文本: {case['text'][:80]}..." if len(case['text']) > 80 else f"    文本: {case['text']}")
        print(f"    真实: {case['true_label']} → 预测: {case['pred_label']} (置信度: {case['confidence']:.3f})")
        print(f"    概率: pos={case['prob_positive']:.3f}, neg={case['prob_negative']:.3f}, neu={case['prob_neutral']:.3f}")
    
    # 保存
    os.makedirs('analysis_results', exist_ok=True)
    df = pd.DataFrame(bad_cases)
    df.to_csv('analysis_results/bad_cases_detailed.csv', index=False, encoding='utf-8')
    print(f"\n✓ Bad cases 已保存到: analysis_results/bad_cases_detailed.csv")
    
    return bad_cases, confusion


# ========== 3. 生成测试集预测 ==========
def generate_test_predictions(model_path, output_file='predictions.txt'):
    """生成测试集预测"""
    
    print("\n" + "="*70)
    print("生成测试集预测")
    print("="*70)
    
    config = BEST_CONFIG
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
    
    # 测试集路径
    test_file = r'D:\当代人工智能\project5\test_without_label.txt'
    
    # 创建测试数据集
    test_dataset = MultimodalDataset(
        data_dir=config['data_dir'],
        split_file=test_file,
        has_label=False
    )
    
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    
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


# ========== 主函数 ==========
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='最终流水线')
    parser.add_argument('--train', action='store_true', help='训练最佳配置模型')
    parser.add_argument('--badcase', action='store_true', help='Bad Case分析')
    parser.add_argument('--predict', action='store_true', help='生成测试集预测')
    parser.add_argument('--all', action='store_true', help='运行全部流程')
    parser.add_argument('--model', type=str, default='experiments/checkpoints/BEST_HP1_cross_attention.pth')
    parser.add_argument('--output', type=str, default='predictions.txt')
    
    args = parser.parse_args()
    
    if args.all:
        # 运行全部流程
        model_path, acc = train_best_model()
        analyze_bad_cases(model_path)
        generate_test_predictions(model_path, args.output)
        print("\n" + "="*70)
        print("✓ 全部流程完成!")
        print("="*70)
    elif args.train:
        train_best_model()
    elif args.badcase:
        analyze_bad_cases(args.model)
    elif args.predict:
        generate_test_predictions(args.model, args.output)
    else:
        print("使用方法:")
        print("  py -3.11 run_final_pipeline.py --all                 # 运行全部")
        print("  py -3.11 run_final_pipeline.py --train               # 仅训练")
        print("  py -3.11 run_final_pipeline.py --badcase             # 仅Bad Case分析")
        print("  py -3.11 run_final_pipeline.py --predict             # 仅生成预测")
        print("  py -3.11 run_final_pipeline.py --badcase --model xxx # 使用指定模型")


if __name__ == '__main__':
    main()
