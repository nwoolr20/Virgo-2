from __future__ import annotations

from virgo2.lm import NeuralFieldLanguageModel


def train_char_field(texts: list[str], epochs: int = 1) -> NeuralFieldLanguageModel:
    model = NeuralFieldLanguageModel()
    model.fit_texts(texts, epochs=epochs)
    return model
