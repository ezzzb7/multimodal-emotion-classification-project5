"""
模型集成预测器 - 快速提升准确率
融合多个已训练模型的预测结果
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from data.data_loader import get_data_loaders
from models.multimodal_model import MultimodalClassifier
from utils.train_utils import set_seed, compute_metrics


class EnsemblePredictor:
    """集成预测器 - 多模型软投票"""
    
    def __init__(self, checkpoint_paths, weights=None, device='cpu'):
        """
        Args:
            checkpoint_paths: 模型检查点路径列表
            weights: 每个模型的权重（默认按验证准确率加权）
            device: 运行设备
        """
        self.device = device
        self.models = []
        self.model_names = []
        
        print("\n" + "="*70)
        print("初始化集成预测器")
        print("="*70)
        
        # 加载所有模型
        for i, ckpt_path in enumerate(checkpoint_paths):
            print(f"\n加载模型 {i+1}/{len(checkpoint_paths)}: {os.path.basename(ckpt_path)}")
            
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            
            # 提取配置
            if 'config' in checkpoint:
                config = checkpoint['config']
            else:
                # 从检查点信息推断配置
                config = {
                    'num_classes': 3,
                    'text_model': checkpoint.get('text_model', 'distilbert-base-uncased'),
                    'image_model': checkpoint.get('image_model', 'resnet50'),
                    'fusion_type': checkpoint.get('fusion_type', 'early'),
                    'feature_dim': checkpoint.get('feature_dim', 512),
                    'freeze_encoders': True,
                    'dropout': checkpoint.get('dropout', 0.3)
                }
            
            # 创建模型
            model = MultimodalClassifier(**config).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.models.append(model)
            
            # 记录模型信息
            model_info = {
                'path': ckpt_path,
                'val_acc': checkpoint.get('val_acc', 0.0),
                'val_f1': checkpoint.get('val_f1', 0.0),
                'fusion_type': config['fusion_type']
            }
            self.model_names.append(model_info)
            
            print(f"  ✓ {config['fusion_type']} fusion")
            print(f"  ✓ Val Acc: {model_info['val_acc']:.4f}, Val F1: {model_info['val_f1']:.4f}")
        
        # 设置权重
        if weights is None:
            # 默认按验证准确率加权
            val_accs = [m['val_acc'] for m in self.model_names]
            weights = np.array(val_accs) / sum(val_accs)
        else:
            weights = np.array(weights) / sum(weights)
        
        self.weights = weights
        
        print(f"\n集成策略:")
        print(f"  - 模型数量: {len(self.models)}")
        print(f"  - 投票方式: 软投票（加权平均）")
        print(f"  - 模型权重:")
        for i, (name, w) in enumerate(zip(self.model_names, self.weights)):
            print(f"    [{i+1}] {name['fusion_type']:20s} weight={w:.4f} (acc={name['val_acc']:.4f})")
        print("="*70 + "\n")
    
    @torch.no_grad()
    def predict(self, texts, images):
        """
        集成预测
        
        Args:
            texts: 文本列表
            images: 图像张量 [B, 3, H, W]
        
        Returns:
            logits: [B, num_classes]
        """
        all_logits = []
        
        for model in self.models:
            logits = model(texts, images)
            all_logits.append(logits)
        
        # 加权平均
        ensemble_logits = sum(w * logits for w, logits in zip(self.weights, all_logits))
        return ensemble_logits
    
    @torch.no_grad()
    def predict_proba(self, texts, images):
        """返回概率分布"""
        logits = self.predict(texts, images)
        return torch.softmax(logits, dim=1)
    
    @torch.no_grad()
    def evaluate(self, data_loader, desc="Evaluating"):
        """
        在数据集上评估集成模型
        
        Args:
            data_loader: 数据加载器
            desc: 进度条描述
        
        Returns:
            metrics: 评估指标字典
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in tqdm(data_loader, desc=desc):
            texts = batch['text']
            images = batch['image'].to(self.device)
            labels = batch['label']
            
            # 集成预测
            probs = self.predict_proba(texts, images)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
        
        # 计算指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        metrics = compute_metrics(all_preds, all_labels)
        
        return metrics, all_preds, all_probs


def main():
    """主函数：评估集成模型"""
    
    # 设置
    set_seed(42)
    device = 'cpu'
    
    # 已有的最佳模型检查点
    checkpoints = [
        'checkpoints/best_early_multimodal_20260120_195503.pth',  # 69%
        'checkpoints/best_cross_attention_multimodal_20260121_021412.pth',  # 68.83%
        'checkpoints/best_late_multimodal_20260120_054159.pth',  # 68.67%
    ]
    
    # 检查文件是否存在
    available_checkpoints = [ckpt for ckpt in checkpoints if os.path.exists(ckpt)]
    
    if len(available_checkpoints) == 0:
        print("❌ 未找到任何可用的检查点文件！")
        print("请确保以下文件存在：")
        for ckpt in checkpoints:
            print(f"  - {ckpt}")
        return
    
    print(f"✓ 找到 {len(available_checkpoints)} 个可用检查点")
    
    # 创建集成预测器
    ensemble = EnsemblePredictor(
        checkpoint_paths=available_checkpoints,
        weights=None,  # 自动按验证准确率加权
        device=device
    )
    
    # 加载验证数据
    print("加载验证数据...")
    _, val_loader, _ = get_data_loaders(
        data_dir='data',
        train_label_file='train.txt',
        test_label_file='test_without_label.txt',
        batch_size=16,
        val_ratio=0.15,
        num_workers=0,
        seed=42,
        img_size=224,
        enhanced_augmentation=False
    )
    
    # 评估单个模型
    print("\n" + "="*70)
    print("单模型性能评估")
    print("="*70)
    
    for i, (model, model_info) in enumerate(zip(ensemble.models, ensemble.model_names)):
        print(f"\n模型 {i+1}: {model_info['fusion_type']}")
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Evaluating", leave=False):
                texts = batch['text']
                images = batch['image'].to(device)
                labels = batch['label']
                
                logits = model(texts, images)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        print(f"  Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    
    # 评估集成模型
    print("\n" + "="*70)
    print("集成模型性能评估")
    print("="*70)
    
    metrics, preds, probs = ensemble.evaluate(val_loader, desc="Evaluating Ensemble")
    
    print(f"\n集成结果:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    # 与最佳单模型对比
    best_single_acc = max(m['val_acc'] for m in ensemble.model_names)
    improvement = metrics['accuracy'] - best_single_acc
    
    print(f"\n提升分析:")
    print(f"  最佳单模型: {best_single_acc:.4f}")
    print(f"  集成模型:   {metrics['accuracy']:.4f}")
    print(f"  提升:       {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    if improvement > 0:
        print(f"\n✅ 集成成功！准确率提升 {improvement*100:.2f}%")
    else:
        print(f"\n⚠️ 集成未带来提升，建议尝试其他权重策略")
    
    print("\n" + "="*70)
    print("评估完成！")
    print("="*70)


if __name__ == "__main__":
    main()
