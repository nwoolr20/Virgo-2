"""SIREN-based neural field for conversation embeddings."""

import torch
import torch.nn as nn
import math
from typing import Dict, List


class SIRENLayer(nn.Module):
    """Single layer with sine activation (SIREN)."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights as per SIREN paper."""
        with torch.no_grad():
            if self.is_first:
                bound = 1 / self.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class ConversationField(nn.Module):
    """
    Neural field mapping 6D coordinates to embedding vectors.
    
    Architecture: SIREN network (sine activations)
    Input: 6D coordinates [0,1]^6
    Output: embedding_dim vector
    """
    
    def __init__(
        self,
        coord_dim: int = 6,
        embedding_dim: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 4,
        omega_0: float = 30.0
    ):
        super().__init__()
        
        self.coord_dim = coord_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Build SIREN network
        layers = []
        layers.append(SIRENLayer(coord_dim, hidden_dim, omega_0, is_first=True))
        
        for _ in range(num_layers - 2):
            layers.append(SIRENLayer(hidden_dim, hidden_dim, omega_0))
            
        self.layers = nn.Sequential(*layers)
        
        # Final linear layer (no sine activation)
        self.final_layer = nn.Linear(hidden_dim, embedding_dim)
        with torch.no_grad():
            bound = math.sqrt(6 / hidden_dim) / omega_0
            self.final_layer.weight.uniform_(-bound, bound)
    
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Query field at given coordinates.
        
        Args:
            coordinates: [batch_size, 6] tensor
            
        Returns:
            embeddings: [batch_size, embedding_dim] tensor
        """
        x = self.layers(coordinates)
        return self.final_layer(x)
    
    def fit_memory(
        self,
        coordinates: torch.Tensor,
        target_embeddings: torch.Tensor,
        num_steps: int = 5000,
        lr: float = 1e-4,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Fit field to memorize coordinate->embedding mappings.
        
        Args:
            coordinates: [N, 6] coordinate tensor
            target_embeddings: [N, embedding_dim] target tensor
            num_steps: Training iterations
            lr: Learning rate
            verbose: Print progress
            
        Returns:
            Training metrics dict
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        
        for step in range(num_steps):
            optimizer.zero_grad()
            predicted = self.forward(coordinates)
            loss = nn.functional.mse_loss(predicted, target_embeddings)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and step % 500 == 0:
                print(f"Step {step}/{num_steps}, Loss: {loss.item():.6f}")
        
        if verbose:
            print(f"Final loss: {losses[-1]:.6f}")
        
        return {"losses": losses}
