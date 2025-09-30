"""Integration tests."""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from virgo import MemorySystem


def test_end_to_end():
    """Complete end-to-end test."""
    system = MemorySystem()
    
    # Simulate conversation
    conversation = [
        ("My name is Alice", 0),
        ("Nice to meet you Alice!", 1),
        ("I have two cats named Whiskers and Mittens", 0),
        ("That's lovely! What are their names again?", 1),
        ("Whiskers and Mittens", 0),
        ("I work as a software engineer", 0),
        ("That sounds interesting!", 1)
    ]
    
    for text, speaker in conversation:
        system.store(text, speaker_id=speaker)
    
    # Train field
    system.fit_field(num_steps=1000, verbose=False)
    
    # Test retrieval
    results = system.retrieve("What is my name?", k=3)
    texts = [m.text for m, _ in results]
    assert any("Alice" in text for text in texts)
    
    results = system.retrieve("Tell me about my pets", k=3)
    texts = [m.text for m, _ in results]
    assert any("cats" in text.lower() or "Whiskers" in text for text in texts)
    
    results = system.retrieve("What do I do for work?", k=3)
    texts = [m.text for m, _ in results]
    assert any("engineer" in text.lower() for text in texts)


def test_persistence_end_to_end():
    """Test complete save/load cycle."""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create and train system
        system1 = MemorySystem()
        
        conversations = [
            ("My favorite color is blue", 0),
            ("I like pizza", 0),
            ("I live in Seattle", 0),
            ("I have a dog named Max", 0)
        ]
        
        for text, speaker in conversations:
            system1.store(text, speaker_id=speaker)
        
        system1.fit_field(num_steps=500, verbose=False)
        system1.save(temp_dir)
        
        # Load and verify
        system2 = MemorySystem()
        system2.load(temp_dir)
        
        # Test queries
        results = system2.retrieve("What color do I like?", k=2)
        texts = [m.text for m, _ in results]
        assert any("blue" in text.lower() for text in texts)
        
        results = system2.retrieve("Where do I live?", k=2)
        texts = [m.text for m, _ in results]
        assert any("Seattle" in text for text in texts)
        
    finally:
        shutil.rmtree(temp_dir)
