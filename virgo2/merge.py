from __future__ import annotations

from .memory import NeuralMemory


def merge_memories(memories: list[NeuralMemory]) -> NeuralMemory:
    if not memories:
        raise ValueError("at least one memory is required")
    base_dim = memories[0].encoder.dimensions
    if any(mem.encoder.dimensions != base_dim for mem in memories):
        raise ValueError("all memories must use the same coordinate dimensions")

    merged = NeuralMemory()
    for memory in memories:
        for record in memory.records:
            merged.add(record.text, metadata=record.metadata, salience=record.salience)
    merged.fit()
    return merged
