from __future__ import annotations

from pathlib import Path

import numpy as np

from virgo2.lm import NeuralFieldLanguageModel


class TextFieldDistiller:
    def __init__(self, seed: int = 0) -> None:
        self.model = NeuralFieldLanguageModel(seed=seed)

    def normalize_text(self, text: str) -> str:
        if text is None:
            raise ValueError("text must not be None")
        cleaned = text.replace("\r\n", "\n").strip()
        if not cleaned:
            raise ValueError("text must not be empty after normalization")
        return cleaned

    def fit_text(self, text: str, epochs: int = 1) -> None:
        normalized = self.normalize_text(text)
        self.model.fit_texts([normalized], epochs=epochs)

    def fit_corpus(self, texts: list[str], epochs: int = 1) -> None:
        if not texts:
            raise ValueError("texts must be non-empty")
        normalized_texts = [self.normalize_text(item) for item in texts]
        self.model.fit_texts(normalized_texts, epochs=epochs)

    def save(self, out_dir: str | Path) -> None:
        self.model.save(out_dir)

    def sample(self, prompt: str, max_chars: int = 120, temperature: float = 0.8) -> str:
        if prompt is None:
            raise ValueError("prompt must not be None")
        return self.model.generate(prompt, max_chars=max_chars, temperature=temperature)

    def evaluate(self, text: str, sample_prompt: str = "") -> dict[str, float | int | bool | str]:
        from virgo2.training.evaluate import evaluate_model

        normalized = self.normalize_text(text)
        return evaluate_model(self.model, normalized, sample_prompt=sample_prompt)

    def validate_checkpoint(self, out_dir: str | Path) -> tuple[bool, str]:
        return NeuralFieldLanguageModel.validate_checkpoint(out_dir)

    def evaluate_checkpoint(self, out_dir: str | Path, text: str, sample_prompt: str = "") -> dict[str, float | int | bool | str]:
        from virgo2.training.evaluate import evaluate_checkpoint

        normalized = self.normalize_text(text)
        return evaluate_checkpoint(out_dir, normalized, sample_prompt=sample_prompt)

    def reconstruct_logits(self, prompt: str) -> np.ndarray:
        if prompt is None:
            raise ValueError("prompt must not be None")
        return self.model.predict_next(prompt)
