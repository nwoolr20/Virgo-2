"""Virgo-2 memory substrate plus experimental DDiF-inspired neural-field LM."""

from .conversation import ConversationMemory, ConversationTurn
from .coordinates import CoordinateEncoder
from .field import NeuralField
from .lifecycle import FieldLifecycleManager
from .memory import MemoryRecord, NeuralMemory
from .registry import FieldInfo, FieldRegistry
from .retrieval import MultiFieldRetriever, RetrievedMemory
from .vault import FieldVault

__all__ = [
    "CoordinateEncoder",
    "NeuralField",
    "MemoryRecord",
    "NeuralMemory",
    "ConversationMemory",
    "ConversationTurn",
    "FieldLifecycleManager",
    "FieldInfo",
    "FieldRegistry",
    "RetrievedMemory",
    "MultiFieldRetriever",
    "FieldVault",
]
