"""Tests for neural field."""

import pytest
import torch
from virgo.field import ConversationField


def test_field_forward():
    """Test basic forward pass."""
    field = ConversationField()
    coords = torch.rand(10, 6)
    embeddings = field(coords)
    
    assert embeddings.shape == (10, 384)


def test_field_overfit():
    """Field should memorize single mapping."""
    field = ConversationField(hidden_dim=128)
    
    coord = torch.rand(1, 6)
    target = torch.rand(1, 384)
    
    metrics = field.fit_memory(coord, target, num_steps=1000, lr=1e-3, verbose=False)
    
    assert metrics["losses"][-1] < 0.01
    
    predicted = field(coord)
    error = torch.norm(predicted - target).item()
    assert error < 0.1


def test_field_batch():
    """Field should handle batches."""
    field = ConversationField()
    
    coords = torch.rand(50, 6)
    targets = torch.randn(50, 384)
    
    metrics = field.fit_memory(coords, targets, num_steps=500, verbose=False)
    
    # Should decrease loss
    assert metrics["losses"][-1] < metrics["losses"][0]
