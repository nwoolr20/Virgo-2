from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .char_codec import CharCodec
from .generation import sample_from_logits


class NeuralFieldLanguageModel:
    def __init__(self, fourier_bands: int = 16, seed: int = 0, context_window: int = 8) -> None:
        self.codec = CharCodec()
        self.fourier_bands = fourier_bands
        self.seed = seed
        self.context_window = context_window
        self.weights: np.ndarray | None = None
        self.training_char_count: int = 0

    @property
    def feature_count(self) -> int:
        return 1 + 2 + (2 * self.fourier_bands * 2)

    def _features(self, coords: np.ndarray) -> np.ndarray:
        bands = np.arange(1, self.fourier_bands + 1, dtype=np.float32)
        x = coords[:, [0]]
        c = coords[:, [1]]
        x_sin = np.sin(2 * np.pi * x * bands)
        x_cos = np.cos(2 * np.pi * x * bands)
        c_sin = np.sin(2 * np.pi * c * bands)
        c_cos = np.cos(2 * np.pi * c * bands)
        return np.concatenate([np.ones((coords.shape[0], 1), dtype=np.float32), x, c, x_sin, x_cos, c_sin, c_cos], axis=1)

    def _context_signature(self, prompt: str) -> float:
        if not prompt:
            return 0.0
        tail = prompt[-self.context_window :]
        weighted = 0
        for i, ch in enumerate(tail):
            weighted += (i + 1) * ord(ch)
        signature_modulus = 10007
        return float(weighted % signature_modulus) / float(signature_modulus - 1)

    def _prediction_coords(self, prompt: str) -> np.ndarray:
        if self.training_char_count > 1:
            denom = float(self.training_char_count - 1)
        else:
            denom = float(max(len(prompt), 1))
        pos_norm = float(len(prompt)) / max(denom, 1.0)
        pos_norm = float(np.clip(pos_norm, 0.0, 1.0))
        return np.array([[pos_norm, self._context_signature(prompt)]], dtype=np.float32)

    def fit_texts(self, texts: list[str], epochs: int = 1) -> None:
        if not texts:
            raise ValueError("texts must be non-empty")
        if epochs != 1:
            raise ValueError(
                "epochs is no longer used for closed-form training; use epochs=1. "
                "This model uses a deterministic closed-form ridge solve."
            )
        corpus = "\n".join(texts)
        self.training_char_count = len(corpus)
        self.codec.fit([corpus])
        ids = self.codec.encode(corpus)
        if len(ids) < 2:
            raise ValueError("Need at least two characters for training")
        n = len(ids) - 1
        coords = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            prompt = corpus[: i + 1]
            coords[i] = self._prediction_coords(prompt)
        X = self._features(coords)
        y = np.array(ids[1:], dtype=np.int64)
        Y = np.eye(self.codec.vocab_size, dtype=np.float32)[y]
        reg = 1e-3
        self.weights = np.linalg.solve(X.T @ X + reg * np.eye(X.shape[1]), X.T @ Y)

    def predict_next(self, prompt: str) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not trained")
        c = self._prediction_coords(prompt)
        return self._features(c) @ self.weights

    def generate(self, prompt: str, max_chars: int = 200, temperature: float = 0.8, seed: int | None = None, deterministic: bool = False) -> str:
        if max_chars < 0:
            raise ValueError("max_chars must be non-negative")
        if self.weights is None:
            raise RuntimeError("Model is not trained")
        if prompt is None:
            raise ValueError("prompt must not be None")
        out = prompt
        rng_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(rng_seed)
        for _ in range(max_chars):
            logits = self.predict_next(out).reshape(-1)
            if deterministic:
                next_id = int(np.argmax(logits))
            else:
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
        meta = {
            "fourier_bands": self.fourier_bands,
            "seed": self.seed,
            "context_window": self.context_window,
            "training_char_count": self.training_char_count,
            "checkpoint_format": "virgo2-fieldlm-v1-json-temporary",
            "migration_todo": "Migrate JSON metadata/codec to stricter typed format in v1.1",
        }
        (p / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NeuralFieldLanguageModel":
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        model = cls(
            fourier_bands=meta["fourier_bands"],
            seed=meta["seed"],
            context_window=meta.get("context_window", 8),
        )
        model.training_char_count = int(meta.get("training_char_count", 0))
        model.weights = np.load(p / "weights.npy")
        model.codec.id_to_char = json.loads((p / "codec.json").read_text(encoding="utf-8"))
        model.codec.char_to_id = {ch: i for i, ch in enumerate(model.codec.id_to_char)}
        return model

    @classmethod
    def validate_checkpoint(cls, path: str | Path) -> tuple[bool, str]:
        p = Path(path)
        required = [p / "weights.npy", p / "meta.json", p / "codec.json"]
        for item in required:
            if not item.exists():
                return False, f"missing required checkpoint file: {item.name}"
        try:
            model = cls.load(p)
        except Exception as exc:
            return False, f"failed to load checkpoint: {exc}"
        if model.weights is None:
            return False, "weights are missing after load"
        expected_features = model.feature_count
        if model.weights.ndim != 2:
            return False, f"weights must be 2D, got ndim={model.weights.ndim}"
        if model.weights.shape[0] != expected_features:
            return False, f"weights feature mismatch: expected {expected_features}, got {model.weights.shape[0]}"
        if model.weights.shape[1] != model.codec.vocab_size:
            return False, f"weights vocab mismatch: expected {model.codec.vocab_size}, got {model.weights.shape[1]}"
        try:
            _ = model.generate("h", max_chars=1, deterministic=True)
        except Exception as exc:
            return False, f"generation smoke test failed: {exc}"
        return True, "ok"
