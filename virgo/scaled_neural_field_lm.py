"""
Scaled Neural Field Language Model with Transformer Integration
100M+ parameter architecture for production-level training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SIRENLayer(nn.Module):
    """SIREN layer with sine activation"""
    
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ScaledCoordinateEncoder(nn.Module):
    """
    Scaled coordinate encoder with larger capacity.
    Maps token sequences to continuous coordinates using transformers.
    """
    
    def __init__(self, vocab_size, coord_dim=8, hidden_dim=512, num_layers=4):
        super().__init__()
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)
        
        # Transformer encoder to process sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Project to coordinates
        self.to_coords = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, coord_dim),
            nn.Sigmoid()  # Keep in [0, 1]^coord_dim
        )
    
    def forward(self, token_ids):
        """
        Args:
            token_ids: [batch, seq_len] token indices
        Returns:
            coords: [batch, seq_len, coord_dim] coordinates
        """
        batch_size, seq_len = token_ids.shape
        
        # Embed tokens
        x = self.token_embed(token_ids)  # [batch, seq_len, hidden_dim]
        
        # Add positional encoding
        if seq_len <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :seq_len, :]
        else:
            # Interpolate positional encoding for longer sequences
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            x = x + pos_enc
        
        # Process with transformer
        h = self.transformer(x)  # [batch, seq_len, hidden_dim]
        
        # Convert to coordinates
        coords = self.to_coords(h)  # [batch, seq_len, coord_dim]
        
        return coords


class ScaledGenerativeField(nn.Module):
    """
    Scaled neural field with larger SIREN network.
    Generates token logits from coordinates.
    """
    
    def __init__(self, coord_dim=8, vocab_size=50257, hidden_dim=1024, num_layers=8):
        super().__init__()
        
        # SIREN network with increased capacity
        layers = [SIRENLayer(coord_dim, hidden_dim, is_first=True)]
        for _ in range(num_layers - 1):
            layers.append(SIRENLayer(hidden_dim, hidden_dim))
        self.siren = nn.Sequential(*layers)
        
        # Output projection to vocabulary
        self.to_logits = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, coordinates):
        """
        Args:
            coordinates: [batch, seq_len, coord_dim]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Query field
        batch, seq_len, coord_dim = coordinates.shape
        coords_flat = coordinates.view(-1, coord_dim)
        
        field_output = self.siren(coords_flat)
        logits = self.to_logits(field_output)
        
        return logits.view(batch, seq_len, -1)
    
    def interpolate(self, coord1, coord2, alpha):
        """
        Interpolate between two coordinates.
        
        Args:
            coord1, coord2: [coord_dim] coordinates
            alpha: interpolation factor [0, 1]
        Returns:
            logits: [vocab_size]
        """
        coord_interp = (1 - alpha) * coord1 + alpha * coord2
        return self.forward(coord_interp.unsqueeze(0).unsqueeze(0))[0, 0]


class ScaledNeuralFieldLM(nn.Module):
    """
    Scaled Neural Field Language Model with 100M+ parameters.
    
    Architecture:
    - Token Embedding (50K vocab, 512 dim)
    - Transformer Encoder (4 layers, 8 heads, 512 dim) 
    - Coordinate Encoder (projects to 8D coordinate space)
    - SIREN Field (8 layers, 1024 dim)
    - Output Layer (1024 -> 50K vocab)
    
    Total parameters: ~100M+ depending on configuration
    """
    
    def __init__(self, vocab_size=50257, coord_dim=8, encoder_hidden_dim=512, 
                 encoder_layers=4, field_hidden_dim=1024, field_layers=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.coord_dim = coord_dim
        
        # Learned coordinate encoder with transformers
        self.coord_encoder = ScaledCoordinateEncoder(
            vocab_size, 
            coord_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_layers
        )
        
        # Generative field with scaled SIREN
        self.field = ScaledGenerativeField(
            coord_dim, 
            vocab_size,
            hidden_dim=field_hidden_dim,
            num_layers=field_layers
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights for better training stability"""
        if isinstance(module, nn.Linear):
            if hasattr(module, 'is_siren') and module.is_siren:
                # SIREN initialization is handled separately
                pass
            else:
                # Standard initialization for other layers
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
        # Encode to coordinates
        coords = self.coord_encoder(input_ids)
        
        # Query field
        logits = self.field(coords)
        
        if target_ids is not None:
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                target_ids.view(-1),
                ignore_index=-100  # Ignore padding tokens
            )
            return logits, loss
        
        return logits
    
    @torch.no_grad()
    def generate(self, prompt_ids, max_length=50, temperature=1.0, top_k=50, top_p=0.95):
        """
        Generate text autoregressively with top-k and top-p sampling.
        
        Args:
            prompt_ids: [seq_len] initial tokens
            max_length: maximum tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling parameter
            top_p: nucleus sampling parameter
        Returns:
            generated_ids: [prompt_len + max_length]
        """
        self.eval()
        generated = prompt_ids.unsqueeze(0)  # [1, seq_len]
        
        for _ in range(max_length):
            # Get logits for current sequence
            logits = self.forward(generated)
            
            # Get logits for last position
            next_logits = logits[0, -1] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = -float('Inf')
            
            # Sample from filtered distribution
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        return generated[0]
    
    @torch.no_grad()
    def interpolate_sequences(self, seq1_ids, seq2_ids, alpha):
        """
        Interpolate between two sequences in coordinate space.
        
        Args:
            seq1_ids, seq2_ids: [seq_len] token sequences
            alpha: interpolation factor [0, 1]
        Returns:
            generated_ids: [seq_len] interpolated sequence
        """
        self.eval()
        
        # Get coordinates for both sequences
        coords1 = self.coord_encoder(seq1_ids.unsqueeze(0))[0]
        coords2 = self.coord_encoder(seq2_ids.unsqueeze(0))[0]
        
        # Interpolate coordinates
        coords_interp = (1 - alpha) * coords1 + alpha * coords2
        
        # Query field at interpolated coordinates
        logits = self.field(coords_interp.unsqueeze(0))[0]
        
        # Greedily decode
        generated = torch.argmax(logits, dim=-1)
        
        return generated
    
    def count_parameters(self):
        """Count total parameters in model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
