"""BPE Tokenizer wrapper for Neural Field Language Model"""

import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer


class BPETokenizer:
    """
    BPE tokenizer using HuggingFace transformers (GPT-2 tokenizer).
    Provides ~50K vocabulary vs ~100 chars for character-level.
    """
    
    def __init__(self, model_name="gpt2"):
        """
        Initialize BPE tokenizer.
        
        Args:
            model_name: HuggingFace model name (default: gpt2 for ~50K vocab)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.vocab_size = len(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        
    def encode(self, text: str, add_eos: bool = True, max_length: Optional[int] = None) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_eos: Whether to add EOS token
            max_length: Maximum sequence length
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if add_eos and tokens[-1] != self.eos_token_id:
            tokens.append(self.eos_token_id)
        
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
            
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def encode_batch(self, texts: List[str], max_length: int = 512, padding: str = "max_length") -> torch.Tensor:
        """
        Encode batch of texts to tensor.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Padding strategy ('max_length' or 'longest')
            
        Returns:
            Tensor of shape [batch_size, max_length]
        """
        encoded = self.tokenizer(
            texts,
            padding=padding,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return encoded["input_ids"]
    
    def decode_batch(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """
        Decode batch of token IDs to texts.
        
        Args:
            token_ids: Tensor of shape [batch_size, seq_len]
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded text strings
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def __len__(self):
        """Return vocabulary size"""
        return self.vocab_size
