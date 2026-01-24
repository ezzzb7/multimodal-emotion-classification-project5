"""
Text Encoder using pretrained DistilBERT
Memory-efficient alternative to full BERT
"""

import torch
import torch.nn as nn
from transformers import (
    DistilBertModel, DistilBertTokenizer,
    RobertaModel, RobertaTokenizer,
    BertModel, BertTokenizer,
    AutoModel, AutoTokenizer
)


class TextEncoder(nn.Module):
    """
    Text encoder using DistilBERT
    Extracts semantic features from text input
    """
    
    def __init__(self, 
                 model_name='distilbert-base-uncased',
                 output_dim=768,
                 freeze_bert=True,
                 max_length=128):
        """
        Args:
            model_name: Pretrained model name
            output_dim: Output feature dimension
            freeze_bert: Whether to freeze BERT parameters
            max_length: Maximum sequence length
        """
        super(TextEncoder, self).__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.output_dim = output_dim
        
        # Load pretrained model (支持多种架构)
        print(f"Loading {model_name}...")
        
        # 根据模型名称选择合适的类
        if 'roberta' in model_name.lower():
            self.bert = RobertaModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                local_files_only=True  # 使用本地缓存
            )
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name, local_files_only=True)
            print(f"  ✓ Using RoBERTa architecture")
        elif 'distilbert' in model_name.lower():
            self.bert = DistilBertModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                local_files_only=True  # 使用本地缓存
            )
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name, local_files_only=True)
            print(f"  ✓ Using DistilBERT architecture")
        elif 'bert' in model_name.lower():
            self.bert = BertModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                local_files_only=True  # 使用本地缓存
            )
            self.tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
            print(f"  ✓ Using BERT architecture")
        else:
            # 使用AutoModel作为后备
            self.bert = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                local_files_only=True  # 使用本地缓存
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            print(f"  ✓ Using Auto architecture")
        
        # Freeze BERT parameters to save memory
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print(f"  ✓ BERT parameters frozen")
        
        # Hidden dimension from DistilBERT
        self.hidden_dim = self.bert.config.hidden_size  # 768
        
        # Projection layer (if output_dim != hidden_dim)
        if output_dim != self.hidden_dim:
            self.projection = nn.Linear(self.hidden_dim, output_dim)
        else:
            self.projection = nn.Identity()
        
        print(f"  ✓ Text encoder initialized: {self.hidden_dim} -> {output_dim}")
    
    def tokenize(self, texts):
        """
        Tokenize text inputs
        
        Args:
            texts: List of text strings
            
        Returns:
            Dict with input_ids and attention_mask
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def forward(self, texts):
        """
        Forward pass
        
        Args:
            texts: List of text strings or dict with tokenized inputs
            
        Returns:
            torch.Tensor: Text features [batch_size, output_dim]
        """
        # If input is list of strings, tokenize first
        if isinstance(texts, list):
            encoded = self.tokenize(texts)
            input_ids = encoded['input_ids'].to(next(self.parameters()).device)
            attention_mask = encoded['attention_mask'].to(next(self.parameters()).device)
        else:
            # Already tokenized
            input_ids = texts['input_ids']
            attention_mask = texts['attention_mask']
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # Project to output dimension
        text_features = self.projection(cls_embedding)  # [batch_size, output_dim]
        
        return text_features
    
    def unfreeze_layers(self, num_layers=2):
        """
        Unfreeze top N layers for fine-tuning
        
        Args:
            num_layers: Number of top layers to unfreeze
        """
        # DistilBERT has 6 transformer layers
        total_layers = len(self.bert.transformer.layer)
        
        for i in range(total_layers - num_layers, total_layers):
            for param in self.bert.transformer.layer[i].parameters():
                param.requires_grad = True
        
        print(f"  ✓ Unfroze top {num_layers} BERT layers")


def test_text_encoder():
    """Test function"""
    print("\n=== Testing Text Encoder ===")
    
    # Create encoder
    encoder = TextEncoder(output_dim=512, freeze_bert=True)
    
    # Test texts
    texts = [
        "I love this beautiful sunset!",
        "This is terrible and disappointing.",
        "It's okay, nothing special."
    ]
    
    # Forward pass
    with torch.no_grad():
        features = encoder(texts)
    
    print(f"Input: {len(texts)} texts")
    print(f"Output shape: {features.shape}")
    print(f"Feature sample: {features[0][:5]}")
    print("✓ Text encoder test passed!\n")


if __name__ == "__main__":
    test_text_encoder()
