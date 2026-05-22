from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(temperature, 1e-6)
    scaled = logits / t
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    return probs / np.sum(probs)


def sample_from_logits(logits: np.ndarray, temperature: float = 0.8, rng: np.random.Generator | None = None) -> int:
    gen = rng or np.random.default_rng(0)
    probs = softmax(logits, temperature)
    return int(gen.choice(np.arange(probs.shape[0]), p=probs))
