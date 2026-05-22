from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .char_codec import CharCodec
from .generation import sample_from_logits


class NeuralFieldLanguageModel:
    def __init__(self, fourier_bands: int = 16, seed: int = 0) -> None:
        self.codec = CharCodec()
        self.fourier_bands = fourier_bands
        self.seed = seed
        self.weights: np.ndarray | None = None

    def _features(self, coords: np.ndarray) -> np.ndarray:
        bands = np.arange(1, self.fourier_bands + 1, dtype=np.float32)
        x = coords[:, [0]]
        sin = np.sin(2 * np.pi * x * bands)
        cos = np.cos(2 * np.pi * x * bands)
        return np.concatenate([np.ones((coords.shape[0], 1), dtype=np.float32), x, sin, cos], axis=1)

    def fit_texts(self, texts: list[str], epochs: int = 200) -> None:
        corpus = "\n".join(texts)
        self.codec.fit([corpus])
        ids = self.codec.encode(corpus)
        if len(ids) < 2:
            raise ValueError("Need at least two characters for training")
        n = len(ids) - 1
        coords = np.arange(n, dtype=np.float32) / max(1, n - 1)
        X = self._features(coords[:, None])
        y = np.array(ids[1:], dtype=np.int64)
        Y = np.eye(self.codec.vocab_size, dtype=np.float32)[y]
        reg = 1e-3
        self.weights = np.linalg.solve(X.T @ X + reg * np.eye(X.shape[1]), X.T @ Y)

    def predict_next(self, prompt: str) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not trained")
        pos = len(prompt)
        c = np.array([[float(pos) / max(pos, 1)]], dtype=np.float32)
        return self._features(c) @ self.weights

    def generate(self, prompt: str, max_chars: int = 200, temperature: float = 0.8) -> str:
        out = prompt
        rng = np.random.default_rng(self.seed)
        for _ in range(max_chars):
            logits = self.predict_next(out).reshape(-1)
            next_id = sample_from_logits(logits, temperature=temperature, rng=rng)
            out += self.codec.decode([next_id])
        return out

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if self.weights is None:
            raise RuntimeError("Model is not trained")
        np.save(p / "weights.npy", self.weights)
        (p / "codec.json").write_text(json.dumps(self.codec.id_to_char), encoding="utf-8")
        (p / "meta.json").write_text(json.dumps({"fourier_bands": self.fourier_bands, "seed": self.seed}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NeuralFieldLanguageModel":
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        model = cls(fourier_bands=meta["fourier_bands"], seed=meta["seed"])
        model.weights = np.load(p / "weights.npy")
        model.codec.id_to_char = json.loads((p / "codec.json").read_text(encoding="utf-8"))
        model.codec.char_to_id = {ch: i for i, ch in enumerate(model.codec.id_to_char)}
        return model
