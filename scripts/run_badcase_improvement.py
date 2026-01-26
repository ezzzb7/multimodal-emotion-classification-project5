"""
基于 Bad Case 分析的针对性改进

主要问题：
1. positive ↔ negative 互相混淆（54%错误）
2. neutral 容易被误分为 positive/negative

改进策略：
1. 增加 positive/negative 边界样本的训练权重
2. 对易混淆样本进行数据增强
3. 使用更细粒度的类别权重
4. 尝试 Focal Loss 处理难样本
5. 对低置信度预测增加惩罚
"""
import os
import sys
import re
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, DistilBertTokenizer
import torchvision.transforms as transforms
from tqdm import tqdm
import random

sys.path.insert(0, os.path.dirname(__file__))

from run_experiment_optimized import OptimizedMultimodalClassifier, OPTIMIZED_CONFIG
from utils.train_utils import set_seed, compute_metrics, EarlyStopping


# ========== 断点续传 ==========
CHECKPOINT_FILE = 'experiments/badcase_improvement_checkpoint.json'


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'results': []}


def save_checkpoint(data):
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# ========== Focal Loss（处理难样本）==========
class FocalLoss(nn.Module):
    """Focal Loss - 降低易分类样本的权重，聚焦难样本"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ========== Label Smoothing Loss ==========
class LabelSmoothingLoss(nn.Module):
    """标签平滑 - 防止过于自信的预测"""
    def __init__(self, num_classes, smoothing=0.1, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)
        
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        smooth_label = one_hot * confidence + (1 - one_hot) * smooth_value
        
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(smooth_label * log_prob).sum(dim=1)
        
        if self.weight is not None:
            weight = self.weight[target]
            loss = loss * weight
        
        return loss.mean()


# ========== 文本清洗 ==========
def clean_text(text):
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


# ========== 数据集（带样本权重）==========
class WeightedDataset(Dataset):
    """支持样本权重的数据集"""
    
    def __init__(self, data_dir, split_file, bad_case_guids=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform or BASIC_TRANSFORM
        self.bad_case_guids = set(bad_case_guids) if bad_case_guids else set()
        
        self.samples = []
        self.sample_weights = []
        self.label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    guid = parts[0]
                    if guid == 'guid':
                        continue
                    label = self.label_map.get(parts[1], 0)
                    self.samples.append((guid, label))
                    
                    # Bad Case 样本权重更高
                    if guid in self.bad_case_guids:
                        self.sample_weights.append(2.0)  # Bad case 权重 x2
                    else:
                        self.sample_weights.append(1.0)
    
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
        
        return {'guid': guid, 'text': text, 'image': image, 'label': label}
    
    def get_sample_weights(self):
        return self.sample_weights


class MultimodalDataset(Dataset):
    """标准数据集"""
    def __init__(self, data_dir, split_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform or BASIC_TRANSFORM
        
        self.samples = []
        self.label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    guid = parts[0]
                    if guid == 'guid':
                        continue
                    label = self.label_map.get(parts[1], 0)
                    self.samples.append((guid, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        guid, label = self.samples[idx]
        
        text_path = os.path.join(self.data_dir, f"{guid}.txt")
        try:
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = clean_text(f.read().strip())
        except:
            text = ""
        
        image_path = os.path.join(self.data_dir, f"{guid}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except:
            image = torch.zeros(3, 224, 224)
        
        return {'guid': guid, 'text': text, 'image': image, 'label': label}


def run_improvement_experiment(exp_id, exp_name, loss_type='ce', class_weights=None, 
                                use_weighted_sampling=False, bad_case_guids=None,
                                focal_gamma=2.0, label_smoothing=0.0):
    """运行单个改进实验"""
    
    print(f"\n{'='*60}")
    print(f"[{exp_id}] {exp_name}")
    print(f"{'='*60}")
    
    config = OPTIMIZED_CONFIG.copy()
    config['dropout'] = 0.2  # HP1最佳配置
    
    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=True)
    
    # 创建数据集
    if use_weighted_sampling and bad_case_guids:
        train_dataset = WeightedDataset(
            data_dir=config['data_dir'],
            split_file='splits/train_split.txt',
            bad_case_guids=bad_case_guids
        )
        # 使用加权采样器
        sampler = WeightedRandomSampler(
            weights=train_dataset.get_sample_weights(),
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                  sampler=sampler, num_workers=0)
    else:
        train_dataset = MultimodalDataset(config['data_dir'], 'splits/train_split.txt')
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                  shuffle=True, num_workers=0)
    
    val_dataset = MultimodalDataset(config['data_dir'], 'splits/val_split.txt')
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=0)
    
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
    
    # 选择损失函数
    if class_weights is not None:
        weight_tensor = torch.FloatTensor(class_weights).to(device)
    else:
        weight_tensor = torch.FloatTensor([1.0, 1.5, 3.0]).to(device)
    
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=weight_tensor, gamma=focal_gamma)
        print(f"使用 Focal Loss (gamma={focal_gamma})")
    elif loss_type == 'label_smoothing':
        criterion = LabelSmoothingLoss(num_classes=3, smoothing=label_smoothing, weight=weight_tensor)
        print(f"使用 Label Smoothing (smoothing={label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print(f"使用 CrossEntropyLoss, 类别权重: {class_weights}")
    
    # 优化器
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
    sys.stdout.flush()
    
    best_val_acc = 0
    best_val_f1 = 0
    best_epoch = 0
    best_model_state = None
    start_time = time.time()
    
    for epoch in range(15):
        model.train()
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
            best_model_state = model.state_dict().copy()
            print(f"    ✓ 新最佳!", flush=True)
        
        if early_stopping(val_metrics['accuracy'], epoch):
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # 保存最佳模型
    if best_model_state:
        os.makedirs('experiments/checkpoints', exist_ok=True)
        torch.save(best_model_state, f'experiments/checkpoints/{exp_id}_best.pth')
    
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
    
    return result


def run_all_improvements():
    """运行所有改进实验"""
    
    print("\n" + "="*70)
    print("基于 Bad Case 的针对性改进实验")
    print("="*70)
    
    # 加载断点
    checkpoint = load_checkpoint()
    completed = set(checkpoint.get('completed', []))
    results = checkpoint.get('results', [])
    
    if completed:
        print(f"✓ 从断点恢复，已完成 {len(completed)} 个实验")
    
    # 加载 Bad Case GUIDs（用于加权采样）
    bad_case_guids = []
    if os.path.exists('analysis_results/bad_cases_detailed.csv'):
        df = pd.read_csv('analysis_results/bad_cases_detailed.csv')
        bad_case_guids = df['guid'].astype(str).tolist()
        print(f"加载 {len(bad_case_guids)} 个 Bad Case")
    
    # 实验配置 - 基于 HP1 最佳配置 (dropout=0.2, lr_classifier=1e-4, weight_decay=0.01)
    # 目标：在 72.25% 基础上进一步提升
    experiments = [
        # (exp_id, exp_name, loss_type, class_weights, weighted_sampling, focal_gamma, label_smoothing)
        ('BC0', 'HP1基线复现', 'ce', [1.0, 1.5, 3.0], False, None, 2.0, 0.0),  # 先复现72.25%基线
        ('BC1', '调整类别权重(1.0,2.0,3.0)', 'ce', [1.0, 2.0, 3.0], False, None, 2.0, 0.0),
        ('BC2', 'Focal Loss(gamma=2)', 'focal', [1.0, 1.5, 3.0], False, None, 2.0, 0.0),
        ('BC3', 'Label Smoothing(0.1)', 'label_smoothing', [1.0, 1.5, 3.0], False, None, 2.0, 0.1),
        ('BC4', 'Bad Case加权采样', 'ce', [1.0, 1.5, 3.0], True, bad_case_guids, 2.0, 0.0),
        ('BC5', 'Focal+调整权重', 'focal', [1.0, 2.0, 3.5], False, None, 2.0, 0.0),
    ]
    
    for exp_id, exp_name, loss_type, class_weights, weighted, bc_guids, gamma, smooth in experiments:
        if exp_id in completed:
            print(f"\n[{exp_id}] 已完成，跳过")
            continue
        
        try:
            result = run_improvement_experiment(
                exp_id=exp_id,
                exp_name=exp_name,
                loss_type=loss_type,
                class_weights=class_weights,
                use_weighted_sampling=weighted,
                bad_case_guids=bc_guids if weighted else None,
                focal_gamma=gamma,
                label_smoothing=smooth
            )
            results.append(result)
            completed.add(exp_id)
            
            save_checkpoint({
                'completed': list(completed),
                'results': results
            })
            print(f"  ✓ 断点已保存")
            
        except Exception as e:
            print(f"  ❌ 实验失败: {e}")
            import traceback
            traceback.print_exc()
            save_checkpoint({
                'completed': list(completed),
                'results': results
            })
            raise
    
    # 打印结果
    print("\n" + "="*70)
    print("Bad Case 改进实验结果")
    print("="*70)
    print(f"{'ID':<8} {'改进策略':<30} {'Val Acc':<10} {'Val F1':<10}")
    print("-"*58)
    
    for r in sorted(results, key=lambda x: -x['val_acc']):
        print(f"{r['exp_id']:<8} {r['exp_name']:<30} {r['val_acc']:.4f}     {r['val_f1']:.4f}")
    
    if results:
        best = max(results, key=lambda x: x['val_acc'])
        print(f"\n🏆 最佳改进策略: {best['exp_name']}")
        print(f"   Val Acc: {best['val_acc']:.4f}")
        print(f"   模型保存: experiments/checkpoints/{best['exp_id']}_best.pth")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Bad Case 改进实验')
    parser.add_argument('--run', action='store_true', help='运行所有改进实验')
    
    args = parser.parse_args()
    
    if args.run:
        run_all_improvements()
    else:
        print("使用方法:")
        print("  py -3.11 run_badcase_improvement.py --run")
