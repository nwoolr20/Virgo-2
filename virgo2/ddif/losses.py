from __future__ import annotations

import numpy as np


def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    idx = np.arange(targets.shape[0])
    return float(-np.log(probs[idx, targets] + 1e-9).mean())
