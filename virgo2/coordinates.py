from __future__ import annotations

import hashlib
import re

import numpy as np


class CoordinateEncoder:
    def __init__(self, dimensions: int = 16, seed: str = "virgo2") -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions
        self.seed = seed
        self._token_re = re.compile(r"\b\w+\b", re.UNICODE)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            return vector
        return vector / norm

    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float64)
        tokens = self._token_re.findall((text or "").lower())
        if not tokens:
            return vector

        freq = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1

        for token, count in freq.items():
            digest = hashlib.blake2b(
                f"{self.seed}:{token}".encode("utf-8"), digest_size=self.dimensions * 2
            ).digest()
            raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float64)
            a = raw[: self.dimensions] / 255.0
            b = raw[self.dimensions : 2 * self.dimensions] / 255.0
            phase = 2.0 * np.pi * a
            freq_band = 1.0 + 4.0 * b
            idx = np.arange(self.dimensions, dtype=np.float64)
            token_vec = np.sin(freq_band * idx + phase) + np.cos((freq_band + 0.5) * idx - phase)
            vector += count * token_vec

        return self._normalize(vector)

    def batch_encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float64)
        return np.vstack([self.encode(text) for text in texts])
