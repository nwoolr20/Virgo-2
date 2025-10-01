"""
True Neural Field Language Model
Generates text by querying a continuous neural field
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


class CoordinateEncoder(nn.Module):
    """
    Learns to map token sequences to continuous coordinates.
    Key: These coordinates are learned to make interpolation meaningful.
    """
    
    def __init__(self, vocab_size, coord_dim=8, hidden_dim=256):
        super().__init__()
        self.coord_dim = coord_dim
        
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        
        # Simple RNN to process sequence
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # Project to coordinates
        self.to_coords = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
        # Embed tokens
        x = self.token_embed(token_ids)  # [batch, seq_len, hidden_dim]
        
        # Process sequence
        h, _ = self.rnn(x)  # [batch, seq_len, hidden_dim]
        
        # Convert to coordinates
        coords = self.to_coords(h)  # [batch, seq_len, coord_dim]
        
        return coords


class GenerativeField(nn.Module):
    """
    Neural field that generates token logits from coordinates.
    This is the core innovation: continuous function approximation.
    """
    
    def __init__(self, coord_dim=8, vocab_size=10000, hidden_dim=512, num_layers=6):
        super().__init__()
        
        # SIREN network
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
        This is what makes neural fields powerful.
        
        Args:
            coord1, coord2: [coord_dim] coordinates
            alpha: interpolation factor [0, 1]
        Returns:
            logits: [vocab_size]
        """
        coord_interp = (1 - alpha) * coord1 + alpha * coord2
        return self.forward(coord_interp.unsqueeze(0).unsqueeze(0))[0, 0]


class NeuralFieldLM(nn.Module):
    """
    Complete neural field language model.
    """
    
    def __init__(self, vocab_size=10000, coord_dim=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.coord_dim = coord_dim
        
        # Learned coordinate encoder
        self.coord_encoder = CoordinateEncoder(vocab_size, coord_dim)
        
        # Generative field
        self.field = GenerativeField(coord_dim, vocab_size)
    
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
                target_ids.view(-1)
            )
            return logits, loss
        
        return logits
    
    @torch.no_grad()
    def generate(self, prompt_ids, max_length=50, temperature=1.0):
        """
        Generate text autoregressively.
        
        Args:
            prompt_ids: [seq_len] initial tokens
            max_length: maximum tokens to generate
            temperature: sampling temperature
        Returns:
            generated_ids: [prompt_len + max_length]
        """
        self.eval()
        generated = prompt_ids.unsqueeze(0)  # [1, seq_len]
        
        for _ in range(max_length):
            # Get logits for current sequence
            logits = self.forward(generated)
            
            # Sample from last position
            next_logits = logits[0, -1] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        return generated[0]
    
    @torch.no_grad()
    def interpolate_sequences(self, seq1_ids, seq2_ids, alpha):
        """
        Interpolate between two sequences in coordinate space.
        This demonstrates the continuous nature of the field.
        
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


def train_neural_field_lm(model, train_data, epochs=10, lr=1e-4, device='cpu'):
    """
    Train the neural field LM on text data.
    
    Args:
        model: NeuralFieldLM
        train_data: list of (input_ids, target_ids) pairs
        epochs: number of training epochs
        lr: learning rate
        device: torch device
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for input_ids, target_ids in train_data:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            _, loss = model(input_ids, target_ids)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
