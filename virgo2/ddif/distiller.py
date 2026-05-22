from __future__ import annotations

from pathlib import Path

import numpy as np

from virgo2.lm import NeuralFieldLanguageModel


class TextFieldDistiller:
    def __init__(self, seed: int = 0) -> None:
        self.model = NeuralFieldLanguageModel(seed=seed)

    def fit_text(self, text: str, epochs: int = 200) -> None:
        self.model.fit_texts([text], epochs=epochs)

    def save(self, out_dir: str | Path) -> None:
        self.model.save(out_dir)

    def sample(self, prompt: str, max_chars: int = 120, temperature: float = 0.8) -> str:
        return self.model.generate(prompt, max_chars=max_chars, temperature=temperature)

    def reconstruct_logits(self, prompt: str) -> np.ndarray:
        return self.model.predict_next(prompt)
