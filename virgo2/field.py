from __future__ import annotations

import numpy as np


class NeuralField:
    def __init__(
        self,
        input_dim: int = 16,
        basis_count: int = 64,
        regularization: float = 1e-4,
        seed: int = 7,
    ) -> None:
        if input_dim <= 0 or basis_count <= 0:
            raise ValueError("input_dim and basis_count must be positive")
        self.input_dim = input_dim
        self.basis_count = basis_count
        self.regularization = regularization
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.frequencies = rng.normal(0, 1, (basis_count, input_dim))
        self.phases = rng.uniform(0, 2 * np.pi, basis_count)
        self.weights: np.ndarray | None = None

    def _validate_coords(self, coordinates: np.ndarray) -> np.ndarray:
        arr = np.asarray(coordinates, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != self.input_dim:
            raise ValueError(f"coordinates must have shape (n, {self.input_dim})")
        return arr

    def features(self, coordinates: np.ndarray) -> np.ndarray:
        coords = self._validate_coords(coordinates)
        projected = coords @ self.frequencies.T + self.phases
        return np.concatenate([np.sin(projected), np.cos(projected)], axis=1)

    def fit(self, coordinates: np.ndarray, targets: np.ndarray) -> None:
        x = self.features(coordinates)
        y = np.asarray(targets, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("coordinates and targets must have matching row counts")
        reg_eye = self.regularization * np.eye(x.shape[1], dtype=np.float64)
        self.weights = np.linalg.solve(x.T @ x + reg_eye, x.T @ y)

    def predict(self, coordinates: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("field is not fitted")
        x = self.features(coordinates)
        return x @ self.weights

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "basis_count": self.basis_count,
            "regularization": self.regularization,
            "seed": self.seed,
            "frequencies": self.frequencies,
            "phases": self.phases,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NeuralField":
        field = cls(
            input_dim=int(data["input_dim"]),
            basis_count=int(data["basis_count"]),
            regularization=float(data["regularization"]),
            seed=int(data["seed"]),
        )
        field.frequencies = np.asarray(data["frequencies"], dtype=np.float64)
        field.phases = np.asarray(data["phases"], dtype=np.float64)
        if data.get("weights") is not None:
            field.weights = np.asarray(data["weights"], dtype=np.float64)
        return field
