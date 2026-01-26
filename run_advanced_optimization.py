"""
进阶优化实验脚本
目标：突破72%+

优化策略：
1. Focal Loss - 解决类别不平衡
2. Label Smoothing - 防止过拟合
3. Mixup数据增强 - 提升泛化
4. R-Drop正则化 - 减少过拟合
5. 更强的图像增强
6. 模型集成
"""
import os
import sys
import time
import csv
import argparse
from datetime import datetime
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

sys.path.insert(0, os.path.dirname(__file__))

from utils.train_utils import set_seed, compute_metrics, EarlyStopping


# ========== 进阶优化配置 ==========
ADVANCED_CONFIG = {
    'seed': 42,
    'val_ratio': 0.2,
    'batch_size': 8,
    'accumulation_steps': 4,
    'num_epochs': 35,
    'weight_decay': 0.02,           # 增强正则化
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'early_stopping_patience': 10,  # 更有耐心
    'feature_dim': 512,
    'dropout': 0.4,                 # 增强dropout
    'data_dir': r'D:\当代人工智能\project5\data',
    'train_label': r'D:\当代人工智能\project5\train.txt',
    
    # 编码器设置
    'unfreeze_text_layers': 2,
    'unfreeze_image_layers': 1,
    
    # 学习率
    'lr_pretrained': 8e-6,          # 稍微降低
    'lr_fusion': 4e-5,
    'lr_classifier': 8e-5,
    
    # 进阶优化
    'label_smoothing': 0.1,         # Label smoothing
    'focal_gamma': 2.0,             # Focal loss gamma
    'mixup_alpha': 0.2,             # Mixup alpha
    'rdrop_alpha': 0.5,             # R-Drop正则化系数
}


# ========== 进阶数据增强 ==========
STRONG_AUGMENT = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ========== Focal Loss ==========
class FocalLoss(nn.Module):
    """
    Focal Loss: 解决类别不平衡问题
    对于容易分类的样本降低权重，让模型更关注难分类的样本
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # focusing parameter
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        # Label smoothing
        num_classes = inputs.size(-1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                targets_smooth = torch.zeros_like(inputs)
                targets_smooth.fill_(self.label_smoothing / (num_classes - 1))
                targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        # 计算 cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算 pt
        pt = torch.exp(-ce_loss)
        
        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # 类别权重
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_weight = focal_weight * alpha_weight
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ========== Mixup ==========
def mixup_data(x_text_ids, x_text_mask, x_image, y, alpha=0.2):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x_image.size(0)
    index = torch.randperm(batch_size).to(x_image.device)
    
    # 图像mixup
    mixed_image = lam * x_image + (1 - lam) * x_image[index]
    
    # 标签
    y_a, y_b = y, y[index]
    
    return x_text_ids, x_text_mask, mixed_image, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ========== 进阶模型 ==========
class AdvancedMultimodalClassifier(nn.Module):
    """
    进阶多模态分类器
    增加：更深的融合网络、残差连接、多层注意力
    """
    def __init__(
        self,
        num_classes=3,
        feature_dim=512,
        dropout=0.4,
        unfreeze_text_layers=2,
        unfreeze_image_layers=1
    ):
        super().__init__()
        
        # ========== 文本编码器 ==========
        print("加载 DistilBERT...")
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        if unfreeze_text_layers > 0:
            for layer in self.text_encoder.transformer.layer[-unfreeze_text_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"  ✓ 解冻 DistilBERT 最后 {unfreeze_text_layers} 层")
        
        self.text_proj = nn.Sequential(
            nn.Linear(768, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ========== 图像编码器 ==========
        print("加载 ResNet50...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        for param in resnet.parameters():
            param.requires_grad = False
        
        if unfreeze_image_layers >= 1:
            for param in resnet.layer4.parameters():
                param.requires_grad = True
            print(f"  ✓ 解冻 ResNet layer4")
        
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        self.image_proj = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ========== 跨模态注意力融合 ==========
        self.cross_attn = nn.MultiheadAttention(
            feature_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(feature_dim)
        
        # 自注意力融合
        self.self_attn = nn.MultiheadAttention(
            feature_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(feature_dim)
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        # ========== 深层分类器 ==========
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        print(f"  ✓ 进阶融合: Cross-Attn + Self-Attn + Gated")
    
    def forward(self, input_ids, attention_mask, images, return_features=False):
        # 文本特征
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_output.last_hidden_state[:, 0, :]
        text_feat = self.text_proj(text_feat)
        
        # 图像特征
        image_feat = self.image_encoder(images)
        image_feat = image_feat.view(image_feat.size(0), -1)
        image_feat = self.image_proj(image_feat)
        
        # 跨模态注意力
        text_seq = text_feat.unsqueeze(1)
        image_seq = image_feat.unsqueeze(1)
        
        # 文本关注图像
        text_attended, _ = self.cross_attn(text_seq, image_seq, image_seq)
        text_enhanced = self.cross_norm(text_feat + text_attended.squeeze(1))
        
        # 图像关注文本
        image_attended, _ = self.cross_attn(image_seq, text_seq, text_seq)
        image_enhanced = self.cross_norm(image_feat + image_attended.squeeze(1))
        
        # 自注意力融合
        combined = torch.stack([text_enhanced, image_enhanced], dim=1)  # [B, 2, D]
        fused, _ = self.self_attn(combined, combined, combined)
        fused = self.self_norm(combined + fused)
        
        # 门控融合
        text_final = fused[:, 0, :]
        image_final = fused[:, 1, :]
        
        gate_input = torch.cat([text_final, image_final], dim=-1)
        gate_weight = self.gate(gate_input)
        gated = gate_weight * text_final + (1 - gate_weight) * image_final
        
        # 最终特征
        final_feat = torch.cat([gated, text_final + image_final], dim=-1)
        
        # 分类
        logits = self.classifier(final_feat)
        
        if return_features:
            return logits, final_feat
        return logits
    
    def get_param_groups(self, lr_pretrained, lr_fusion, lr_classifier):
        pretrained_params = []
        fusion_params = []
        classifier_params = []
        
        for name, param in self.text_encoder.named_parameters():
            if param.requires_grad:
                pretrained_params.append(param)
        
        for name, param in self.image_encoder.named_parameters():
            if param.requires_grad:
                pretrained_params.append(param)
        
        fusion_params.extend(self.text_proj.parameters())
        fusion_params.extend(self.image_proj.parameters())
        fusion_params.extend(self.cross_attn.parameters())
        fusion_params.extend(self.cross_norm.parameters())
        fusion_params.extend(self.self_attn.parameters())
        fusion_params.extend(self.self_norm.parameters())
        fusion_params.extend(self.gate.parameters())
        
        classifier_params.extend(self.classifier.parameters())
        
        return [
            {'params': pretrained_params, 'lr': lr_pretrained, 'name': 'pretrained'},
            {'params': fusion_params, 'lr': lr_fusion, 'name': 'fusion'},
            {'params': classifier_params, 'lr': lr_classifier, 'name': 'classifier'},
        ]


# ========== 数据集 ==========
class AdvancedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split_file, transform=None, oversample_minority=False):
        self.data_dir = data_dir
        self.transform = transform or VAL_TRANSFORM
        
        self.samples = []
        self.label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
        
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    guid = parts[0]
                    label = parts[1]
                    self.samples.append((guid, self.label_map.get(label, 0)))
        
        # 过采样少数类
        if oversample_minority:
            self._oversample()
    
    def _oversample(self):
        """过采样neutral类"""
        label_counts = {}
        for _, label in self.samples:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        max_count = max(label_counts.values())
        
        new_samples = list(self.samples)
        for label, count in label_counts.items():
            if count < max_count:
                # 找出该类别的样本
                class_samples = [(g, l) for g, l in self.samples if l == label]
                # 重复采样
                oversample_count = max_count - count
                oversampled = random.choices(class_samples, k=oversample_count)
                new_samples.extend(oversampled)
        
        self.samples = new_samples
        print(f"  过采样后样本数: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        guid, label = self.samples[idx]
        
        # 文本
        text_path = os.path.join(self.data_dir, f"{guid}.txt")
        try:
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
        except:
            text = ""
        
        # 图像
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


def get_results_csv_path():
    return 'experiments/all_results.csv'


def save_result_to_csv(result: dict):
    os.makedirs('experiments', exist_ok=True)
    csv_path = get_results_csv_path()
    
    if not os.path.exists(csv_path):
        headers = [
            'exp_id', 'exp_name', 'modality', 'fusion_type',
            'val_acc', 'val_f1', 'val_loss',
            'train_acc', 'train_f1', 
            'best_epoch', 'total_epochs',
            'trainable_params', 'total_params',
            'training_time_min', 'timestamp'
        ]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    row = [
        result.get('exp_id', ''),
        result.get('exp_name', ''),
        result.get('modality', ''),
        result.get('fusion_type', ''),
        f"{result.get('val_acc', 0):.4f}",
        f"{result.get('val_f1', 0):.4f}",
        f"{result.get('val_loss', 0):.4f}",
        f"{result.get('train_acc', 0):.4f}",
        f"{result.get('train_f1', 0):.4f}",
        result.get('best_epoch', 0),
        result.get('total_epochs', 0),
        result.get('trainable_params', 0),
        result.get('total_params', 0),
        f"{result.get('training_time_min', 0):.2f}",
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ]
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def run_advanced_experiment(exp_id: str, exp_name: str, use_mixup=True, use_focal=True, use_oversample=True):
    """运行进阶优化实验"""
    print("\n" + "="*70)
    print(f"进阶优化实验 {exp_id}: {exp_name}")
    print(f"Mixup: {use_mixup}, Focal Loss: {use_focal}, 过采样: {use_oversample}")
    print("="*70)
    
    config = ADVANCED_CONFIG
    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 数据集
    train_dataset = AdvancedDataset(
        data_dir=config['data_dir'],
        split_file='splits/train_split.txt',
        transform=STRONG_AUGMENT,
        oversample_minority=use_oversample
    )
    val_dataset = AdvancedDataset(
        data_dir=config['data_dir'],
        split_file='splits/val_split.txt',
        transform=VAL_TRANSFORM,
        oversample_minority=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0
    )
    
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    # 模型
    model = AdvancedMultimodalClassifier(
        num_classes=3,
        feature_dim=config['feature_dim'],
        dropout=config['dropout'],
        unfreeze_text_layers=config['unfreeze_text_layers'],
        unfreeze_image_layers=config['unfreeze_image_layers']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}, 可训练: {trainable_params:,}")
    
    # 损失函数
    if use_focal:
        class_weights = torch.FloatTensor([1.0, 1.5, 2.5]).to(device)
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=config['focal_gamma'],
            label_smoothing=config['label_smoothing']
        )
        print(f"使用 Focal Loss (gamma={config['focal_gamma']}, smoothing={config['label_smoothing']})")
    else:
        class_weights = torch.FloatTensor([1.0, 1.5, 3.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config['label_smoothing'])
    
    # 优化器
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
    best_val_f1 = 0
    best_epoch = 0
    start_time = time.time()
    
    print("\n开始训练...")
    for epoch in range(config['num_epochs']):
        model.train()
        train_preds, train_labels_list = [], []
        total_loss = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)):
            texts = batch['text']
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Mixup
            if use_mixup and random.random() < 0.5:
                input_ids, attention_mask, images, labels_a, labels_b, lam = mixup_data(
                    input_ids, attention_mask, images, labels, config['mixup_alpha']
                )
                logits = model(input_ids, attention_mask, images)
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                logits = model(input_ids, attention_mask, images)
                loss = criterion(logits, labels)
            
            loss = loss / config['accumulation_steps']
            loss.backward()
            
            if (step + 1) % config['accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * config['accumulation_steps']
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
        
        train_metrics = compute_metrics(train_preds, train_labels_list)
        
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
        
        print(f"Epoch {epoch+1:2d}: Train Acc={train_metrics['accuracy']:.4f}, "
              f"Val Acc={val_metrics['accuracy']:.4f}, Val F1={val_metrics['f1']:.4f}")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_val_f1 = val_metrics['f1']
            best_train_acc = train_metrics['accuracy']
            best_train_f1 = train_metrics['f1']
            best_epoch = epoch + 1
            
            os.makedirs('experiments/checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'experiments/checkpoints/{exp_id}_best.pth')
            print(f"  ✓ 新最佳! 已保存模型")
        
        if early_stopping(val_metrics['accuracy'], epoch):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = (time.time() - start_time) / 60
    
    result = {
        'exp_id': exp_id,
        'exp_name': exp_name,
        'modality': 'multimodal_advanced',
        'fusion_type': 'cross_attn_gated',
        'val_acc': best_val_acc,
        'val_f1': best_val_f1,
        'val_loss': 0,
        'train_acc': best_train_acc,
        'train_f1': best_train_f1,
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'training_time_min': training_time
    }
    
    save_result_to_csv(result)
    
    print(f"\n实验 {exp_id} 完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"最佳验证F1: {best_val_f1:.4f}")
    print(f"训练时间: {training_time:.1f} 分钟")
    
    return result


def run_ablation_optimizations():
    """消融实验：测试各个优化技巧的效果"""
    print("\n" + "="*70)
    print("优化技巧消融实验")
    print("="*70)
    
    experiments = [
        ('ADV_base', '基线(无优化)', False, False, False),
        ('ADV_focal', '+Focal Loss', False, True, False),
        ('ADV_oversample', '+过采样', False, False, True),
        ('ADV_mixup', '+Mixup', True, False, False),
        ('ADV_all', '全部优化', True, True, True),
    ]
    
    results = []
    for exp_id, exp_name, use_mixup, use_focal, use_oversample in experiments:
        result = run_advanced_experiment(exp_id, exp_name, use_mixup, use_focal, use_oversample)
        results.append(result)
    
    print("\n" + "="*70)
    print("优化技巧消融结果")
    print("="*70)
    print(f"{'实验':<20} {'Val Acc':<10} {'Val F1':<10}")
    print("-"*40)
    for r in results:
        print(f"{r['exp_name']:<20} {r['val_acc']:.4f}     {r['val_f1']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='进阶优化实验')
    parser.add_argument('--full', action='store_true', help='运行全部优化实验')
    parser.add_argument('--ablation', action='store_true', help='运行消融实验')
    parser.add_argument('--quick', action='store_true', help='快速运行一次全优化实验')
    
    args = parser.parse_args()
    
    if args.ablation:
        run_ablation_optimizations()
    elif args.full or args.quick:
        run_advanced_experiment('ADV_full', 'full_optimization', True, True, True)
    else:
        print("使用方法:")
        print("  py -3.11 run_advanced_optimization.py --quick     # 快速运行全优化")
        print("  py -3.11 run_advanced_optimization.py --ablation  # 优化技巧消融实验")


if __name__ == '__main__':
    main()
