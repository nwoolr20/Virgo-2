"""Simple character-level tokenizer for NFLM"""

import torch
from typing import List, Dict


class CharTokenizer:
    """Character-level tokenizer for simple text generation"""
    
    def __init__(self):
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size = 0
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.eos_token = '<EOS>'
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        # Special tokens
        special_tokens = [self.pad_token, self.unk_token, self.eos_token]
        chars = set(special_tokens)
        
        # Collect all unique characters
        for text in texts:
            chars.update(text)
        
        # Build mappings
        chars = sorted(list(chars))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
    
    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        """Convert text to token IDs"""
        tokens = []
        for char in text:
            tokens.append(self.char_to_idx.get(char, self.char_to_idx[self.unk_token]))
        
        if add_eos:
            tokens.append(self.char_to_idx[self.eos_token])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs to text"""
        chars = []
        for idx in token_ids:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                if char == self.eos_token:
                    break
                if char not in [self.pad_token, self.unk_token]:
                    chars.append(char)
        
        return ''.join(chars)
    
    def encode_batch(self, texts: List[str], max_length: int = None) -> torch.Tensor:
        """Encode batch of texts to tensor"""
        encoded = [self.encode(text) for text in texts]
        
        # Determine max length
        if max_length is None:
            max_length = max(len(seq) for seq in encoded)
        
        # Pad sequences
        padded = []
        for seq in encoded:
            if len(seq) < max_length:
                seq = seq + [self.char_to_idx[self.pad_token]] * (max_length - len(seq))
            else:
                seq = seq[:max_length]
            padded.append(seq)
        
        return torch.tensor(padded, dtype=torch.long)
    
    def decode_batch(self, token_ids: torch.Tensor) -> List[str]:
        """Decode batch of token IDs to texts"""
        return [self.decode(ids.tolist()) for ids in token_ids]
