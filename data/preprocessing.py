"""
Text and image preprocessing utilities
Enhanced version with stronger augmentation
"""

import re
import random
import torch
from torchvision import transforms
from PIL import Image


# 常见缩写词扩展字典
CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am",
    "i've": "i have", "isn't": "is not", "it's": "it is", "let's": "let us",
    "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "we'd": "we would", "we'll": "we will",
    "we're": "we are", "we've": "we have", "weren't": "were not",
    "what's": "what is", "won't": "will not", "wouldn't": "would not",
    "you'd": "you would", "you'll": "you will", "you're": "you are",
    "you've": "you have", "gonna": "going to", "wanna": "want to",
    "gotta": "got to", "kinda": "kind of", "sorta": "sort of"
}

# Emoji情感映射 (简化版)
EMOJI_SENTIMENT = {
    # 正面
    ':)': ' happy ', ':-)': ' happy ', ':D': ' very happy ', ':-D': ' very happy ',
    ';)': ' wink ', ';-)': ' wink ', ':P': ' playful ', ':-P': ' playful ',
    '<3': ' love ', ':*': ' kiss ', '^^': ' happy ',
    # 负面
    ':(': ' sad ', ':-(': ' sad ', ":'(": ' crying ', ":')": ' tears ',
    '>:(': ' angry ', '>:-(': ' angry ', ':/': ' skeptical ', ':-/': ' skeptical ',
    # 中性
    ':o': ' surprised ', ':-o': ' surprised ', ':O': ' shocked ', ':-O': ' shocked ',
}


class TextPreprocessor:
    """Enhanced text preprocessing for sentiment classification"""
    
    def __init__(self, remove_emoji=False, lowercase=True, remove_hashtags=False,
                 expand_contractions=True, convert_emoji=True, mode='train'):
        """
        Args:
            remove_emoji: Whether to remove emojis completely
            lowercase: Whether to convert text to lowercase
            remove_hashtags: Whether to remove hashtag symbols (#)
            expand_contractions: Whether to expand contractions (don't -> do not)
            convert_emoji: Whether to convert text emoticons to words
            mode: 'train' or 'val'/'test'
        """
        self.remove_emoji = remove_emoji
        self.lowercase = lowercase
        self.remove_hashtags = remove_hashtags
        self.expand_contractions = expand_contractions
        self.convert_emoji = convert_emoji
        self.mode = mode
    
    def _expand_contractions(self, text):
        """Expand contractions in text"""
        for contraction, expansion in CONTRACTIONS.items():
            text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
        return text
    
    def _convert_emoticons(self, text):
        """Convert text emoticons to sentiment words"""
        for emoticon, sentiment in EMOJI_SENTIMENT.items():
            text = text.replace(emoticon, sentiment)
        return text
    
    def _process_hashtags(self, text):
        """Process hashtags: #HelloWorld -> hello world"""
        # 保留hashtag内容但分词 (驼峰命名分割)
        def split_hashtag(match):
            tag = match.group(1)
            # 分割驼峰命名
            words = re.sub(r'([A-Z])', r' \1', tag).strip()
            return ' ' + words.lower() + ' '
        
        text = re.sub(r'#(\w+)', split_hashtag, text)
        return text
    
    def clean_text(self, text):
        """
        Clean and normalize text
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if text is None or text.strip() == '':
            return ""
        
        # Convert text emoticons to words (before removing special chars)
        if self.convert_emoji:
            text = self._convert_emoticons(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Handle emojis (Unicode)
        if self.remove_emoji:
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r'', text)
        
        # Process hashtags (split camelCase)
        if not self.remove_hashtags:
            text = self._process_hashtags(text)
        else:
            text = re.sub(r'#', '', text)
        
        # Expand contractions
        if self.expand_contractions:
            text = self._expand_contractions(text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        return text
    
    def __call__(self, text):
        """Make the preprocessor callable"""
        return self.clean_text(text)


class ImagePreprocessor:
    """Image preprocessing with enhanced augmentation for training"""
    
    def __init__(self, mode='train', img_size=224, enhanced=False):
        """
        Args:
            mode: 'train', 'val', or 'test' - determines augmentation strategy
            img_size: Target image size (default: 224 for ResNet/ViT)
            enhanced: Use enhanced augmentation (stronger transforms)
        """
        self.mode = mode
        self.img_size = img_size
        
        if mode == 'train':
            if enhanced:
                # ⭐ 增强版数据增强流水线
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),  # 随机旋转
                    transforms.ColorJitter(
                        brightness=0.3, 
                        contrast=0.3, 
                        saturation=0.3,
                        hue=0.15
                    ),
                    transforms.RandomGrayscale(p=0.1),  # 随机灰度
                    transforms.RandomPerspective(distortion_scale=0.2, p=0.2),  # 透视变换
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))  # 随机擦除
                ])
            else:
                # 标准数据增强流水线
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(img_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.2, 
                        contrast=0.2, 
                        saturation=0.2,
                        hue=0.1
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
        else:  # val or test
            # Simple preprocessing without augmentation
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __call__(self, image):
        """
        Preprocess image
        
        Args:
            image: PIL Image or image path string
            
        Returns:
            torch.Tensor: Preprocessed image tensor [3, H, W]
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image or str, got {type(image)}")
        
        return self.transform(image)


def get_text_preprocessor(mode='train', enhanced=False, **kwargs):
    """Factory function to get text preprocessor"""
    if enhanced:
        # 增强版默认参数
        kwargs.setdefault('expand_contractions', True)
        kwargs.setdefault('convert_emoji', True)
    return TextPreprocessor(mode=mode, **kwargs)


def get_image_preprocessor(mode='train', img_size=224, enhanced=False):
    """Factory function to get image preprocessor"""
    return ImagePreprocessor(mode=mode, img_size=img_size, enhanced=enhanced)

