from __future__ import annotations

import numpy as np


def normalized_position(idx: int, length: int) -> float:
    if length <= 1:
        return 0.0
    return float(idx) / float(length - 1)


def context_hash(text: str, mod: int = 997) -> float:
    return (sum(ord(c) for c in text) % mod) / float(mod)


def coordinate_vector(doc_id: int, pos: int, length: int, context: str = "", scale: float = 1.0) -> np.ndarray:
    return np.array(
        [float(doc_id), normalized_position(pos, length), context_hash(context), float(scale)],
        dtype=np.float32,
    )
