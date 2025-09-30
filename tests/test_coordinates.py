"""Tests for coordinate system."""

import pytest
import torch
from virgo.coordinates import ConversationCoordinateSystem


def test_coordinate_dimensions():
    """All coordinates should be in [0,1]."""
    system = ConversationCoordinateSystem()
    coords = system.extract_coordinates(
        text="My name is Alice",
        timestamp=1234567890.0,
        turn_id=1,
        speaker_id=0
    )
    
    assert coords.shape == (6,)
    assert torch.all((coords >= 0) & (coords <= 1))


def test_importance_scoring():
    """Test importance heuristics."""
    system = ConversationCoordinateSystem()
    
    high = system.compute_importance("What is my name? I'm worried.")
    low = system.compute_importance("The weather is nice.")
    
    assert high > low
    assert 0 <= high <= 1
    assert 0 <= low <= 1


def test_semantic_consistency():
    """Similar texts should cluster."""
    system = ConversationCoordinateSystem()
    
    texts = [
        "I love programming",
        "Coding is my passion",
        "The weather is nice",
        "It's sunny outside"
    ]
    system.fit_semantic_projection(texts)
    
    coord1 = system.extract_coordinates("I enjoy coding", 1.0, 1, 0)
    coord2 = system.extract_coordinates("Programming is fun", 1.0, 2, 0)
    coord3 = system.extract_coordinates("The sky is blue", 1.0, 3, 0)
    
    semantic_diff_prog = abs(coord1[2] - coord2[2])
    semantic_diff_weather = abs(coord1[2] - coord3[2])
    
    assert semantic_diff_prog < semantic_diff_weather
