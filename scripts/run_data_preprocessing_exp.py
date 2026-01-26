"""
完整的数据预处理与增强实验
按照实验要求：数据预处理上对文本进行清洗、对图片进行增强

实验设计：
1. 基线：无任何预处理
2. 文本清洗：URL移除、@mentions移除、特殊字符处理
3. 图像增强：RandomCrop、ColorJitter、RandomHorizontalFlip
4. 全部应用：文本清洗 + 图像增强
"""
import os
import sys
import re
import time
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, DistilBertTokenizer
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from run_experiment_optimized import OptimizedMultimodalClassifier, OPTIMIZED_CONFIG
from utils.train_utils import set_seed, compute_metrics, EarlyStopping


# ========== 断点续传 ==========
CHECKPOINT_FILE = 'experiments/data_aug_checkpoint.json'


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'results': []}


def save_checkpoint(data):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# ========== 文本清洗函数 ==========
def clean_text_basic(text):
    """基础文本清洗"""
    if not isinstance(text, str):
        return ""
    return text.strip()


def clean_text_advanced(text):
    """高级文本清洗"""
    if not isinstance(text, str):
        return ""
    
    # 移除URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # 移除@mentions
    text = re.sub(r'@\w+', '', text)
    
    # 保留#hashtag的文字部分
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符（保留基本标点）
    text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
    
    return text.strip()


# ========== 图像Transform ==========
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
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证用transform（始终无增强）
VAL_TRANSFORM = BASIC_TRANSFORM


# ========== 数据集 ==========
class PreprocessedDataset(Dataset):
    """支持不同预处理方式的数据集"""
    
    def __init__(self, data_dir, split_file, image_transform, text_clean_fn):
        self.data_dir = data_dir
        self.image_transform = image_transform
        self.text_clean_fn = text_clean_fn
        
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
        
        # 应用文本清洗
        text = self.text_clean_fn(text)
        
        # 加载图像
        image_path = os.path.join(self.data_dir, f"{guid}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)
        except:
            image = torch.zeros(3, 224, 224)
        
        return {
            'guid': guid,
            'text': text,
            'image': image,
            'label': label
        }


def run_preprocessing_experiment(exp_id, exp_name, train_img_transform, text_clean_fn, 
                                  use_best_hyperparams=True):
    """运行单个预处理实验"""
    
    print(f"\n{'='*60}")
    print(f"[{exp_id}] {exp_name}")
    print(f"{'='*60}")
    
    # 使用最佳超参数（HP1: dropout=0.2）
    config = OPTIMIZED_CONFIG.copy()
    if use_best_hyperparams:
        config['dropout'] = 0.2  # HP1最佳配置
    
    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 加载tokenizer
    print("加载 Tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=True)
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = PreprocessedDataset(
        data_dir=config['data_dir'],
        split_file='splits/train_split.txt',
        image_transform=train_img_transform,
        text_clean_fn=text_clean_fn
    )
    val_dataset = PreprocessedDataset(
        data_dir=config['data_dir'],
        split_file='splits/val_split.txt',
        image_transform=VAL_TRANSFORM,  # 验证集不增强
        text_clean_fn=text_clean_fn  # 但文本清洗保持一致
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=0)
    
    print(f"  训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    # 创建模型
    print("创建模型...")
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
    
    total_steps = len(train_loader) * 15 // config['accumulation_steps']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    early_stopping = EarlyStopping(patience=7, mode='max')
    
    # 训练
    print("\n开始训练...")
    import sys
    sys.stdout.flush()
    
    best_val_acc = 0
    best_val_f1 = 0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(15):
        model.train()
        batch_count = 0
        print(f"  Epoch {epoch+1}/15 [", end="", flush=True)
        
        for step, batch in enumerate(train_loader):
            texts = batch['text']
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            encoded = tokenizer(list(texts), padding=True, truncation=True, 
                              max_length=128, return_tensors='pt')
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
            
            batch_count += 1
            if batch_count % 20 == 0:
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
                
                encoded = tokenizer(list(texts), padding=True, truncation=True,
                                  max_length=128, return_tensors='pt')
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                logits = model(input_ids, attention_mask, images)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_metrics = compute_metrics(val_preds, val_labels_list)
        print(f"Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}", flush=True)
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            
            os.makedirs('experiments/checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'experiments/checkpoints/{exp_id}_best.pth')
            print(f"    ✓ 新最佳!", flush=True)
        
        if early_stopping(val_metrics['accuracy'], epoch):
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    training_time = (time.time() - start_time) / 60
    
    result = {
        'exp_id': exp_id,
        'exp_name': exp_name,
        'val_acc': best_val_acc,
        'val_f1': best_val_f1,
        'best_epoch': best_epoch,
        'training_time': training_time
    }
    
    print(f"\n实验 {exp_id} 完成!")
    print(f"  最佳 Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"  训练时间: {training_time:.1f} 分钟")
    
    return result


def run_all_preprocessing_experiments():
    """运行所有预处理实验"""
    
    print("\n" + "="*70)
    print("数据预处理与增强对比实验")
    print("="*70)
    
    # 加载断点
    checkpoint = load_checkpoint()
    completed = set(checkpoint.get('completed', []))
    results = checkpoint.get('results', [])
    
    if completed:
        print(f"✓ 从断点恢复，已完成 {len(completed)} 个实验")
    
    experiments = [
        # (exp_id, exp_name, train_img_transform, text_clean_fn)
        ('DA1', '基线(无预处理)', BASIC_TRANSFORM, clean_text_basic),
        ('DA2', '仅文本清洗', BASIC_TRANSFORM, clean_text_advanced),
        ('DA3', '仅图像增强', AUGMENTED_TRANSFORM, clean_text_basic),
        ('DA4', '文本清洗+图像增强', AUGMENTED_TRANSFORM, clean_text_advanced),
    ]
    
    for exp_id, exp_name, img_transform, text_fn in experiments:
        if exp_id in completed:
            print(f"\n[{exp_id}] 已完成，跳过")
            continue
        
        try:
            result = run_preprocessing_experiment(exp_id, exp_name, img_transform, text_fn)
            results.append(result)
            completed.add(exp_id)
            
            save_checkpoint({
                'completed': list(completed),
                'results': results
            })
            print(f"  ✓ 断点已保存")
            
        except Exception as e:
            print(f"  ❌ 实验失败: {e}")
            save_checkpoint({
                'completed': list(completed),
                'results': results
            })
            raise
    
    # 打印结果
    print("\n" + "="*70)
    print("数据预处理实验结果")
    print("="*70)
    print(f"{'ID':<8} {'预处理方式':<25} {'Val Acc':<10} {'Val F1':<10}")
    print("-"*53)
    
    for r in sorted(results, key=lambda x: -x['val_acc']):
        print(f"{r['exp_id']:<8} {r['exp_name']:<25} {r['val_acc']:.4f}     {r['val_f1']:.4f}")
    
    if results:
        best = max(results, key=lambda x: x['val_acc'])
        print(f"\n🏆 最佳预处理方式: {best['exp_name']}")
        print(f"   Val Acc: {best['val_acc']:.4f}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='数据预处理实验')
    parser.add_argument('--run', action='store_true', help='运行所有预处理实验')
    
    args = parser.parse_args()
    
    if args.run:
        run_all_preprocessing_experiments()
    else:
        print("使用方法:")
        print("  py -3.11 run_data_preprocessing_exp.py --run")
