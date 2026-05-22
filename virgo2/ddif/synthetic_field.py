from __future__ import annotations

import numpy as np


def reconstruct_distribution(coords: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return coords @ weights
