"""
多模态数据集类
加载配对的文本和图像数据
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class MultimodalDataset(Dataset):
    """多模态情感分类数据集"""
    
    def __init__(self, 
                 data_dir,
                 label_file,
                 text_preprocessor=None,
                 image_preprocessor=None,
                 mode='train'):
        """
        Args:
            data_dir: 数据目录路径，包含所有的.txt和.jpg文件
            label_file: 标签文件路径，格式为 guid,label
            text_preprocessor: 文本预处理器对象
            image_preprocessor: 图像预处理器对象
            mode: 'train', 'val', 'test'
        """
        self.data_dir = data_dir
        self.mode = mode
        
        # 读取标签文件
        self.df = pd.read_csv(label_file, sep=',', header=None, names=['guid', 'tag'])
        
        # 标签映射：根据附件中的图片，tag列可能包含positive/negative/neutral或null
        self.label_map = {
            'positive': 0, 
            'neutral': 1, 
            'negative': 2
        }
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        
        # 预处理器
        self.text_preprocessor = text_preprocessor
        self.image_preprocessor = image_preprocessor
        
        print(f"[{mode.upper()}] Loaded {len(self.df)} samples from {label_file}")
        if 'tag' in self.df.columns:
            label_counts = self.df['tag'].value_counts()
            print(f"Label distribution: {label_counts.to_dict()}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'guid': str,
                'text': str (preprocessed),
                'image': torch.Tensor (C, H, W),
                'label': int (-1 for test without label)
            }
        """
        row = self.df.iloc[idx]
        guid = row['guid']
        
        # === 加载文本 ===
        txt_path = os.path.join(self.data_dir, f"{guid}.txt")
        try:
            # Try multiple encodings to handle various text files
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            text = None
            for encoding in encodings:
                try:
                    with open(txt_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            
            if text is None:
                # Fallback: read as binary and decode with errors='replace'
                with open(txt_path, 'rb') as f:
                    text = f.read().decode('utf-8', errors='replace')
            
            if self.text_preprocessor:
                text = self.text_preprocessor(text)
        except FileNotFoundError:
            print(f"Warning: Text file not found for guid {guid}")
            text = "missing text"
        except Exception as e:
            print(f"Warning: Error loading text {guid}: {e}")
            text = "error text"
        
        # === 加载图像 ===
        img_path = os.path.join(self.data_dir, f"{guid}.jpg")
        try:
            image = Image.open(img_path).convert('RGB')
            if self.image_preprocessor:
                image = self.image_preprocessor(image)
            else:
                # 默认转换为tensor
                from torchvision import transforms
                image = transforms.ToTensor()(image)
        except FileNotFoundError:
            print(f"Warning: Image file not found for guid {guid}")
            # 返回黑色图像
            image = torch.zeros(3, 224, 224)
        except Exception as e:
            print(f"Warning: Error loading image {guid}: {e}")
            image = torch.zeros(3, 224, 224)
        
        # === 处理标签 ===
        tag = row['tag']
        if pd.isna(tag) or str(tag).lower() == 'null':
            label = -1  # 测试集无标签
        else:
            tag_lower = str(tag).lower()
            label = self.label_map.get(tag_lower, -1)
            if label == -1:
                print(f"Warning: Unknown label '{tag}' for guid {guid}")
        
        return {
            'guid': guid,
            'text': text,
            'image': image,
            'label': label
        }
    
    def get_label_name(self, label_idx):
        """将标签索引转换为标签名称"""
        return self.inv_label_map.get(label_idx, 'unknown')
