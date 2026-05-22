import numpy as np

from virgo2.coordinates import CoordinateEncoder


def test_same_text_same_coords() -> None:
    enc = CoordinateEncoder(dimensions=16)
    assert np.allclose(enc.encode("hello world"), enc.encode("hello world"))


def test_different_text_different_coords() -> None:
    enc = CoordinateEncoder()
    assert not np.allclose(enc.encode("alpha"), enc.encode("beta"))


def test_dimension_and_norm() -> None:
    enc = CoordinateEncoder(dimensions=24)
    vec = enc.encode("virgo memory")
    assert vec.shape == (24,)
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_batch_encode() -> None:
    enc = CoordinateEncoder(dimensions=8)
    arr = enc.batch_encode(["a", "b", "c"])
    assert arr.shape == (3, 8)
