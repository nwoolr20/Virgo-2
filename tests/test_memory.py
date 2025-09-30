"""Tests for memory system."""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from virgo.memory import MemorySystem


def test_store_retrieve():
    """Test basic store and retrieve."""
    system = MemorySystem()
    
    system.store("My name is Alice", speaker_id=0)
    system.store("Nice to meet you!", speaker_id=1)
    system.store("I like tea", speaker_id=0)
    
    results = system.retrieve("What is my name?", k=2)
    
    assert len(results) == 2
    texts = [m.text for m, _ in results]
    assert any("Alice" in text for text in texts)


def test_field_training():
    """Test field training."""
    system = MemorySystem()
    
    for i in range(10):
        system.store(f"Message {i}", speaker_id=i % 2)
    
    metrics = system.fit_field(num_steps=500, verbose=False)
    
    assert system.is_fitted
    assert metrics["losses"][-1] < metrics["losses"][0]


def test_persistence():
    """THE CRITICAL TEST: Persistence across restart."""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Phase 1: Store and save
        system1 = MemorySystem()
        system1.store("My name is Alice", speaker_id=0)
        system1.store("I prefer tea over coffee", speaker_id=0)
        system1.store("Nice to meet you!", speaker_id=1)
        
        system1.fit_field(num_steps=1000, verbose=False)
        system1.save(temp_dir)
        
        # Phase 2: Load in new instance
        system2 = MemorySystem()
        system2.load(temp_dir)
        
        # Verify data persisted
        assert len(system2.memories) == 3
        assert system2.is_fitted
        
        # Test retrieval still works
        results = system2.retrieve("What is my name?", k=2)
        texts = [m.text for m, _ in results]
        assert any("Alice" in text for text in texts)
        
    finally:
        shutil.rmtree(temp_dir)


def test_stats():
    """Test statistics."""
    system = MemorySystem()
    
    system.store("User message 1", speaker_id=0)
    system.store("Assistant message 1", speaker_id=1)
    system.store("User message 2", speaker_id=0)
    
    stats = system.get_stats()
    
    assert stats["total_memories"] == 3
    assert stats["user_turns"] == 2
    assert stats["assistant_turns"] == 1
    assert stats["is_fitted"] == False
