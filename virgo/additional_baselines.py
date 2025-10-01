"""
Additional Baseline Models for Comparison

Implements various baseline architectures to compare against neural field LM:
1. LSTM Baseline - Recurrent architecture
2. Transformer + MLP - Test SIREN value
3. Coordinate Transformer - Test coordinate value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMBaseline(nn.Module):
    """
    LSTM-based language model baseline.
    Tests against sequential processing without attention.
    """
    
    def __init__(self, vocab_size=50257, embedding_dim=512, hidden_dim=1024, num_layers=4, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
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
            target_ids: [batch, seq_len] (optional)
        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: scalar (if target_ids provided)
        """
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        # LSTM forward
        x, _ = self.lstm(x)
        
        x = self.layer_norm(x)
        logits = self.output(x)
        
        if target_ids is not None:
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
        hidden = None
        
        for _ in range(max_length):
            # Get logits for current sequence
            x = self.embedding(generated)
            x, hidden = self.lstm(x, hidden)
            x = self.layer_norm(x)
            logits = self.output(x)
            
            # Get logits for last position
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
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        return generated[0]
    
    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerMLPBaseline(nn.Module):
    """
    Transformer encoder + MLP decoder (no coordinates, no SIREN).
    Tests if coordinate encoding and SIREN specifically help.
    """
    
    def __init__(self, vocab_size=50257, d_model=512, nhead=8, num_encoder_layers=4, 
                 mlp_hidden_dim=1024, mlp_layers=4, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token + position embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # MLP decoder (instead of SIREN field)
        mlp_layers_list = []
        for i in range(mlp_layers):
            if i == 0:
                mlp_layers_list.append(nn.Linear(d_model, mlp_hidden_dim))
            else:
                mlp_layers_list.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim))
            mlp_layers_list.append(nn.GELU())
            mlp_layers_list.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*mlp_layers_list)
        self.output = nn.Linear(mlp_hidden_dim, vocab_size)
        
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
            target_ids: [batch, seq_len] (optional)
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
        
        # Apply MLP decoder
        x = self.mlp(x)
        logits = self.output(x)
        
        if target_ids is not None:
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


class CoordinateTransformerBaseline(nn.Module):
    """
    Coordinate encoder + Transformer decoder (no SIREN).
    Tests if coordinate encoding helps even without SIREN activation.
    """
    
    def __init__(self, vocab_size=50257, coord_dim=8, d_model=512, nhead=8, 
                 num_encoder_layers=4, num_decoder_layers=6, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.coord_dim = coord_dim
        
        # Import coordinate encoder from scaled model
        from virgo.scaled_neural_field_lm import ScaledCoordinateEncoder
        
        # Coordinate encoder (same as neural field)
        self.coord_encoder = ScaledCoordinateEncoder(
            vocab_size,
            coord_dim,
            hidden_dim=d_model,
            num_layers=num_encoder_layers
        )
        
        # Transformer decoder (instead of SIREN field)
        # Projects from coord_dim to d_model for processing
        self.coord_projection = nn.Linear(coord_dim, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, target_ids=None):
        """
        Args:
            input_ids: [batch, seq_len]
            target_ids: [batch, seq_len] (optional)
        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: scalar (if target_ids provided)
        """
        # Encode to coordinates
        coords = self.coord_encoder(input_ids)  # [batch, seq_len, coord_dim]
        
        # Project to model dimension
        x = self.coord_projection(coords)  # [batch, seq_len, d_model]
        
        # Apply transformer decoder (self-attention)
        x = self.transformer_decoder(x, x)
        
        # Output projection
        x = self.ln_f(x)
        logits = self.output(x)
        
        if target_ids is not None:
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
