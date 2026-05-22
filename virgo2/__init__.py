"""Virgo-2 memory substrate plus experimental DDiF-inspired neural-field LM."""

from .coordinates import CoordinateEncoder
from .field import NeuralField
from .memory import MemoryRecord, NeuralMemory

__all__ = ["CoordinateEncoder", "NeuralField", "MemoryRecord", "NeuralMemory"]
