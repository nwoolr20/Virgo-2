from __future__ import annotations

import numpy as np

from virgo2.lm.char_codec import CharCodec


class TextCoordinateSet:
    def __init__(self, codec: CharCodec) -> None:
        self.codec = codec

    def from_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        ids = self.codec.encode(text)
        n = len(ids)
        if n == 0:
            return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        coords = (np.arange(n, dtype=np.float32) / max(1, n - 1)).reshape(-1, 1)
        return coords, np.array(ids, dtype=np.int64)
