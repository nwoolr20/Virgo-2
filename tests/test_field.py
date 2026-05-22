import numpy as np
import pytest

from virgo2.field import NeuralField


def test_fit_and_predict() -> None:
    f = NeuralField(input_dim=4, basis_count=8)
    x = np.random.default_rng(0).normal(size=(10, 4))
    y = np.random.default_rng(1).normal(size=(10, 1))
    f.fit(x, y)
    pred = f.predict(x)
    assert pred.shape == (10, 1)


def test_bad_dimensions_raise() -> None:
    f = NeuralField(input_dim=4)
    with pytest.raises(ValueError):
        f.features(np.ones((2, 3)))


def test_predict_before_fit_raises() -> None:
    f = NeuralField(input_dim=4)
    with pytest.raises(RuntimeError):
        f.predict(np.ones((1, 4)))


def test_serialization_roundtrip() -> None:
    f = NeuralField(input_dim=4, basis_count=8)
    x = np.random.default_rng(0).normal(size=(5, 4))
    y = np.random.default_rng(0).normal(size=(5, 1))
    f.fit(x, y)
    rt = NeuralField.from_dict(f.to_dict())
    assert np.allclose(f.predict(x), rt.predict(x))
