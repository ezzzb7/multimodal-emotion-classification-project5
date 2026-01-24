"""
针对Bad Case的数据增强策略
基于错误分析结果，对难样本进行针对性增强
"""
import random
import re
from typing import List, Dict
import pandas as pd
import numpy as np


class BadCaseAugmenter:
    """针对Bad Case的数据增强器"""
    
    def __init__(self, bad_cases_csv='analysis_results/bad_cases.csv'):
        """
        Args:
            bad_cases_csv: bad case分析结果CSV文件
        """
        self.bad_cases = pd.read_csv(bad_cases_csv) if bad_cases_csv else None
        
        # 情感词典（用于同义词替换，保持情感倾向）
        self.positive_words = {
            'good': ['great', 'excellent', 'wonderful', 'amazing', 'fantastic'],
            'like': ['love', 'enjoy', 'appreciate', 'adore'],
            'happy': ['joyful', 'delighted', 'pleased', 'cheerful'],
            'best': ['finest', 'greatest', 'top', 'superior'],
            'beautiful': ['gorgeous', 'stunning', 'lovely', 'attractive']
        }
        
        self.negative_words = {
            'bad': ['awful', 'terrible', 'horrible', 'poor'],
            'hate': ['dislike', 'despise', 'detest'],
            'sad': ['unhappy', 'depressed', 'miserable', 'sorrowful'],
            'worst': ['poorest', 'weakest', 'inferior'],
            'ugly': ['unattractive', 'hideous', 'unsightly']
        }
    
    def augment_text(self, text: str, label: str, methods: List[str] = None) -> List[str]:
        """
        对文本进行增强
        
        Args:
            text: 原始文本
            label: 情感标签 (positive/negative/neutral)
            methods: 增强方法列表，默认使用所有方法
        
        Returns:
            augmented_texts: 增强后的文本列表
        """
        if methods is None:
            methods = ['synonym', 'insert', 'delete', 'swap']
        
        augmented = []
        
        for method in methods:
            if method == 'synonym':
                aug_text = self.synonym_replacement(text, label)
                if aug_text != text:
                    augmented.append(aug_text)
            
            elif method == 'insert':
                aug_text = self.random_insertion(text, label)
                augmented.append(aug_text)
            
            elif method == 'delete':
                aug_text = self.random_deletion(text)
                augmented.append(aug_text)
            
            elif method == 'swap':
                aug_text = self.random_swap(text)
                augmented.append(aug_text)
        
        return augmented
    
    def synonym_replacement(self, text: str, label: str, n: int = 2) -> str:
        """同义词替换（保持情感倾向）"""
        # 处理非字符串文本
        if not isinstance(text, str) or not text.strip():
            return text if isinstance(text, str) else ""
        
        words = text.split()
        
        # 选择合适的同义词词典
        if label == 'positive':
            synonym_dict = self.positive_words
        elif label == 'negative':
            synonym_dict = self.negative_words
        else:
            return text  # neutral不做替换
        
        # 随机替换n个词
        replaced = 0
        for i in range(len(words)):
            word_lower = words[i].lower()
            if word_lower in synonym_dict and replaced < n:
                synonyms = synonym_dict[word_lower]
                words[i] = random.choice(synonyms)
                replaced += 1
        
        return ' '.join(words)
    
    def random_insertion(self, text: str, label: str, n: int = 1) -> str:
        """随机插入情感词"""
        words = text.split()
        
        # 选择情感词
        if label == 'positive':
            insert_words = ['really', 'very', 'so', 'absolutely', 'definitely']
        elif label == 'negative':
            insert_words = ['really', 'very', 'so', 'absolutely', 'totally']
        else:
            return text
        
        for _ in range(n):
            insert_word = random.choice(insert_words)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, insert_word)
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """随机删除词（不删除情感词）"""
        words = text.split()
        
        if len(words) == 1:
            return text
        
        # 保护的情感词
        protected = {'good', 'bad', 'great', 'terrible', 'love', 'hate', 
                    'like', 'dislike', 'best', 'worst', 'not', 'no'}
        
        new_words = []
        for word in words:
            if word.lower() not in protected and random.random() > p:
                new_words.append(word)
            else:
                new_words.append(word)
        
        if len(new_words) == 0:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """随机交换词序"""
        words = text.split()
        
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def augment_bad_cases(self, output_file: str, augment_factor: int = 3, data_dir: str = 'data', min_confidence: float = 0.0):
        """
        对bad cases进行数据增强
        
        Args:
            output_file: 输出文件路径
            augment_factor: 每个bad case增强的倍数（默认3倍，降低噪声）
            data_dir: 数据目录，用于检查文件是否存在
            min_confidence: 最小置信度阈值，只增强高置信度错误（0.0=全部，0.7=高置信度）
        """
        if self.bad_cases is None:
            print("⚠️ No bad cases loaded!")
            return
        
        augmented_data = []
        skipped_count = 0
        filtered_by_confidence = 0
        
        for _, row in self.bad_cases.iterrows():
            # 置信度过滤（使用confidence字段）
            if 'confidence' in row and row['confidence'] < min_confidence:
                filtered_by_confidence += 1
                continue
                
            guid = row['guid']
            text = row['text']
            label = row['true_label']
            
            # 检查文件是否存在
            txt_path = os.path.join(data_dir, f"{guid}.txt")
            img_path = os.path.join(data_dir, f"{guid}.jpg")
            
            if not os.path.exists(txt_path) or not os.path.exists(img_path):
                skipped_count += 1
                continue
            
            # 跳过空文本或无效数据
            if pd.isna(text) or not isinstance(text, str) or len(str(text).strip()) == 0:
                skipped_count += 1
                continue
            
            # 确保text是字符串
            text = str(text)
            
            # 原始样本
            augmented_data.append({
                'guid': guid,
                'text': text,
                'tag': label,
                'source': 'original_bad_case'
            })
            
            # 增强样本（重用原始GUID以复用图像文件）
            for i in range(augment_factor):
                aug_texts = self.augment_text(text, label)
                for j, aug_text in enumerate(aug_texts):
                    augmented_data.append({
                        'guid': guid,  # 重用原始GUID，复用图像文件
                        'text': aug_text,
                        'tag': label,
                        'source': f'augmented_bad_case_method_{j}'
                    })
        
        # 保存（使用制表符分隔，避免文本中的逗号干扰）
        if not augmented_data:
            print("⚠️ No valid samples to save!")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in augmented_data:
                f.write(f"{sample['guid']}\t{sample['text']}\t{sample['tag']}\n")
        
        valid_originals = len([d for d in augmented_data if d['source'] == 'original_bad_case'])
        
        print(f"\n增强统计:")
        print(f"  原始bad cases: {len(self.bad_cases)}")
        if min_confidence > 0:
            print(f"  置信度过滤: {filtered_by_confidence} (阈值>{min_confidence:.2f})")
        print(f"  跳过样本: {skipped_count} (文件缺失或无效)")
        print(f"  有效原始样本: {valid_originals}")
        print(f"  增强后总样本: {len(augmented_data)}")
        print(f"  增强样本数: {len(augmented_data) - valid_originals}")
        print(f"  增强倍率: {(len(augmented_data) / valid_originals):.1f}x" if valid_originals > 0 else "0x")
        print(f"✓ 已保存到: {output_file}")


class ImprovedTextPreprocessor:
    """改进的文本预处理器"""
    
    def __init__(self):
        # Emoji情感映射
        self.emoji_sentiment = {
            '😊': ' happy ', '😃': ' happy ', '😁': ' happy ', '🙂': ' happy ',
            '😢': ' sad ', '😭': ' sad ', '😞': ' sad ',
            '😡': ' angry ', '😠': ' angry ',
            '❤️': ' love ', '💕': ' love ', '💖': ' love ',
            '👍': ' good ', '👎': ' bad ',
            '😍': ' love ', '🥰': ' love ',
            '🤔': ' thinking ', '😕': ' confused ',
            '😂': ' laugh ', '🤣': ' laugh ',
            '🔥': ' amazing ', '⭐': ' great ',
        }
        
        # 缩写扩展
        self.contractions = {
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "haven't": "have not", "hasn't": "has not",
            "hadn't": "had not", "won't": "will not", "wouldn't": "would not",
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "can't": "cannot", "couldn't": "could not", "shouldn't": "should not",
            "mightn't": "might not", "mustn't": "must not",
            "i'm": "i am", "you're": "you are", "he's": "he is",
            "she's": "she is", "it's": "it is", "we're": "we are",
            "they're": "they are", "i've": "i have", "you've": "you have",
            "we've": "we have", "they've": "they have",
            "i'd": "i would", "you'd": "you would", "he'd": "he would",
            "she'd": "she would", "we'd": "we would", "they'd": "they would",
            "i'll": "i will", "you'll": "you will", "he'll": "he will",
            "she'll": "she will", "we'll": "we will", "they'll": "they will",
        }
    
    def preprocess(self, text: str) -> str:
        """改进的预处理流程"""
        # 1. 转emoji为情感词
        text = self.convert_emoji_to_sentiment(text)
        
        # 2. 扩展缩写
        text = self.expand_contractions(text)
        
        # 3. 清理特殊字符（保留重要标点）
        text = self.clean_text(text)
        
        # 4. 处理重复字符
        text = self.reduce_lengthening(text)
        
        return text
    
    def convert_emoji_to_sentiment(self, text: str) -> str:
        """将emoji转换为情感词"""
        for emoji, sentiment in self.emoji_sentiment.items():
            text = text.replace(emoji, sentiment)
        return text
    
    def expand_contractions(self, text: str) -> str:
        """扩展英文缩写"""
        text_lower = text.lower()
        for contraction, expansion in self.contractions.items():
            text_lower = text_lower.replace(contraction, expansion)
        return text_lower
    
    def clean_text(self, text: str) -> str:
        """清理文本（保留重要标点）"""
        # 移除URL
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # 移除@mentions
        text = re.sub(r'@\w+', '', text)
        
        # 保留重要标点：!?.,
        text = re.sub(r'[^\w\s!?.,]', ' ', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def reduce_lengthening(self, text: str) -> str:
        """减少重复字符 (goooood -> good)"""
        # 保留最多2个重复字符
        pattern = re.compile(r'(.)\1{2,}')
        return pattern.sub(r'\1\1', text)


def main():
    """演示用法"""
    print("="*70)
    print("Bad Case数据增强工具")
    print("="*70)
    
    # 1. 分析bad cases（需要先运行 analyze_bad_cases.py）
    print("\n步骤1: 确保已运行 bad case分析")
    print("  运行: python analyze_bad_cases.py")
    
    # 2. 对bad cases进行增强
    print("\n步骤2: 对bad cases进行数据增强")
    try:
        augmenter = BadCaseAugmenter('analysis_results/bad_cases.csv')
        augmenter.augment_bad_cases(
            output_file='data/augmented_bad_cases.txt',
            augment_factor=2,  # 每个bad case增强2倍（避免过拟合）
            data_dir=r'D:\当代人工智能\project5\data',  # 实际数据目录
            min_confidence=0.7  # 只增强高置信度错误
        )
    except FileNotFoundError:
        print("⚠️ 未找到bad_cases.csv，请先运行 analyze_bad_cases.py")
        print("演示增强效果:")
        
        # 演示
        augmenter = BadCaseAugmenter(bad_cases_csv=None)
        demo_text = "This movie is really good and I like it"
        print(f"\n原始: {demo_text}")
        print("增强结果:")
        for i, aug in enumerate(augmenter.augment_text(demo_text, 'positive'), 1):
            print(f"  {i}. {aug}")
    
    # 3. 演示改进的预处理
    print("\n步骤3: 改进的文本预处理")
    preprocessor = ImprovedTextPreprocessor()
    
    demo_texts = [
        "I looooove this!!! 😍😍😍",
        "It's soooo bad 😭 I can't believe it",
        "Check out this link: http://example.com @user",
    ]
    
    for text in demo_texts:
        processed = preprocessor.preprocess(text)
        print(f"  原始: {text}")
        print(f"  处理: {processed}\n")


if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)
    os.makedirs('analysis_results', exist_ok=True)
    main()
