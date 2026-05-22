from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coordinates import CoordinateEncoder
from .field import NeuralField


@dataclass
class MemoryRecord:
    text: str
    metadata: dict[str, str] | None = None
    salience: float = 1.0


class NeuralMemory:
    def __init__(self, encoder: CoordinateEncoder | None = None, field: NeuralField | None = None) -> None:
        self.encoder = encoder or CoordinateEncoder()
        self.field = field or NeuralField(input_dim=self.encoder.dimensions)
        if self.field.input_dim != self.encoder.dimensions:
            raise ValueError("encoder dimensions and field input_dim must match")
        self.records: list[MemoryRecord] = []
        self.coordinates = np.zeros((0, self.encoder.dimensions), dtype=np.float64)

    def add(self, text: str, metadata: dict[str, str] | None = None, salience: float = 1.0) -> MemoryRecord:
        if not text.strip():
            raise ValueError("text must be non-empty")
        if salience < 0:
            raise ValueError("salience must be non-negative")
        record = MemoryRecord(text=text, metadata=metadata, salience=salience)
        self.records.append(record)
        self.coordinates = np.vstack([self.coordinates, self.encoder.encode(text)])
        return record

    def fit(self) -> None:
        if not self.records:
            return
        targets = np.array([[r.salience] for r in self.records], dtype=np.float64)
        self.field.fit(self.coordinates, targets)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[MemoryRecord, float]]:
        if not self.records:
            return []
        query_vec = self.encoder.encode(query)
        q = query_vec.reshape(1, -1)
        cos_sim = self.coordinates @ query_vec
        base_distance = 1.0 - cos_sim

        resonance = np.zeros(len(self.records), dtype=np.float64)
        if self.field.weights is not None:
            pred = float(self.field.predict(q)[0, 0])
            saliences = np.array([r.salience for r in self.records], dtype=np.float64)
            resonance = np.abs(saliences - pred)

        salience_boost = np.array([r.salience for r in self.records], dtype=np.float64)
        score = base_distance + 0.2 * resonance - 0.15 * salience_boost

        top_idx = np.argsort(score)[: max(0, k)]
        return [(self.records[i], float(score[i])) for i in top_idx]

    def update_salience(self, index: int, salience: float) -> None:
        if index < 0 or index >= len(self.records):
            raise IndexError("record index out of range")
        if salience < 0:
            raise ValueError("salience must be non-negative")
        self.records[index].salience = salience

    def decay(self, rate: float = 0.01) -> None:
        if not 0 <= rate <= 1:
            raise ValueError("rate must be between 0 and 1")
        for record in self.records:
            record.salience *= 1.0 - rate

    def to_bundle(self) -> dict:
        return {
            "encoder": {"dimensions": self.encoder.dimensions, "seed": self.encoder.seed},
            "field": self.field.to_dict(),
            "records": [
                {"text": r.text, "metadata": r.metadata or {}, "salience": r.salience} for r in self.records
            ],
            "coordinates": self.coordinates,
        }

    @classmethod
    def from_bundle(cls, bundle: dict) -> "NeuralMemory":
        encoder_cfg = bundle["encoder"]
        memory = cls(
            encoder=CoordinateEncoder(
                dimensions=int(encoder_cfg["dimensions"]),
                seed=str(encoder_cfg["seed"]),
            ),
            field=NeuralField.from_dict(bundle["field"]),
        )
        memory.records = [
            MemoryRecord(text=item["text"], metadata=item.get("metadata"), salience=float(item["salience"]))
            for item in bundle["records"]
        ]
        memory.coordinates = np.asarray(bundle["coordinates"], dtype=np.float64)
        return memory
