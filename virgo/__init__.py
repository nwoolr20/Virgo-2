"""Virgo Neural Field Language Model."""

from .coordinates import ConversationCoordinateSystem
from .field import ConversationField, SIRENLayer
from .memory import MemorySystem, Memory
from .chat import SimpleChat
from .neural_field_lm import (
    NeuralFieldLM,
    CoordinateEncoder,
    GenerativeField,
    train_neural_field_lm
)
from .tokenizer import CharTokenizer

__version__ = "0.1.0"

__all__ = [
    "ConversationCoordinateSystem",
    "ConversationField",
    "SIRENLayer",
    "MemorySystem",
    "Memory",
    "SimpleChat",
    "NeuralFieldLM",
    "CoordinateEncoder",
    "GenerativeField",
    "train_neural_field_lm",
    "CharTokenizer",
]
