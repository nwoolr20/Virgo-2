#!/usr/bin/env python3
"""
Baseline Transformer for Comparison with Neural Field LM

This implements a standard transformer language model with the same parameter count
as the neural field model, to test the compression hypothesis.

Phase 3: The Key Experiment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BaselineTransformerLM(nn.Module):
    """
    Standard transformer language model for comparison.
    Configured to match neural field LM parameter count (~98M).
    """
    
    def __init__(self, vocab_size=50257, d_model=512, nhead=8, num_layers=12, 
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token + Position embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embed.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, target_ids=None):
        """
        Args:
            input_ids: [batch, seq_len]
            target_ids: [batch, seq_len] (optional, for training)
        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: scalar (if target_ids provided)
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        token_emb = self.token_embed(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(pos_ids)
        
        x = token_emb + pos_emb
        
        # Apply transformer
        x = self.transformer(x)
        
        # Apply final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if target_ids is not None:
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                target_ids.view(-1),
                ignore_index=-100
            )
            return logits, loss
        
        return logits
    
    @torch.no_grad()
    def generate(self, prompt_ids, max_length=50, temperature=1.0, top_k=50, top_p=0.95):
        """Generate text autoregressively"""
        self.eval()
        generated = prompt_ids.unsqueeze(0)
        
        for _ in range(max_length):
            logits = self.forward(generated)
            next_logits = logits[0, -1] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        return generated[0]
    
    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
