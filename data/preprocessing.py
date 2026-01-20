"""
Text and image preprocessing utilities
"""

import re
import torch
from torchvision import transforms
from PIL import Image


class TextPreprocessor:
    """Text preprocessing for sentiment classification"""
    
    def __init__(self, remove_emoji=False, lowercase=True, remove_hashtags=False):
        """
        Args:
            remove_emoji: Whether to remove emojis completely
            lowercase: Whether to convert text to lowercase
            remove_hashtags: Whether to remove hashtag symbols (#)
        """
        self.remove_emoji = remove_emoji
        self.lowercase = lowercase
        self.remove_hashtags = remove_hashtags
    
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
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Handle emojis
        if self.remove_emoji:
            # Remove emoji characters (basic approach)
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE)
            text = emoji_pattern.sub(r'', text)
        
        # Handle hashtags
        if self.remove_hashtags:
            text = re.sub(r'#', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\#\@]', ' ', text)
        
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
    """Image preprocessing with augmentation for training"""
    
    def __init__(self, mode='train', img_size=224):
        """
        Args:
            mode: 'train', 'val', or 'test' - determines augmentation strategy
            img_size: Target image size (default: 224 for ResNet/ViT)
        """
        self.mode = mode
        self.img_size = img_size
        
        if mode == 'train':
            # Training augmentation pipeline
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
                    mean=[0.485, 0.456, 0.406],  # ImageNet stats
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


def get_text_preprocessor(mode='train', **kwargs):
    """Factory function to get text preprocessor"""
    return TextPreprocessor(**kwargs)


def get_image_preprocessor(mode='train', img_size=224):
    """Factory function to get image preprocessor"""
    return ImagePreprocessor(mode=mode, img_size=img_size)

