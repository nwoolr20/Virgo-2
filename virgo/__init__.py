"""Virgo Neural Field Language Model."""

from .field import ConversationField, SIRENLayer
from .neural_field_lm import (
    NeuralFieldLM,
    CoordinateEncoder,
    GenerativeField,
    train_neural_field_lm
)
from .tokenizer import CharTokenizer
from .bpe_tokenizer import BPETokenizer
from .scaled_neural_field_lm import (
    ScaledNeuralFieldLM,
    ScaledCoordinateEncoder,
    ScaledGenerativeField
)
from .baseline_transformer import BaselineTransformerLM
from .additional_baselines import (
    LSTMBaseline,
    TransformerMLPBaseline,
    CoordinateTransformerBaseline
)

__version__ = "0.1.0"

__all__ = [
    "ConversationField",
    "SIRENLayer",
    "NeuralFieldLM",
    "CoordinateEncoder",
    "GenerativeField",
    "train_neural_field_lm",
    "CharTokenizer",
    "BPETokenizer",
    "ScaledNeuralFieldLM",
    "ScaledCoordinateEncoder",
    "ScaledGenerativeField",
    "BaselineTransformerLM",
    "LSTMBaseline",
    "TransformerMLPBaseline",
    "CoordinateTransformerBaseline",
]
