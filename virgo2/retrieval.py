from __future__ import annotations

from dataclasses import dataclass

from .memory import MemoryRecord, NeuralMemory


@dataclass
class RetrievedMemory:
    field_name: str
    record: MemoryRecord
    score: float
    rank: int


class MultiFieldRetriever:
    def __init__(self, fields: dict[str, NeuralMemory]) -> None:
        self.fields = fields

    def search(self, query: str, k: int = 5) -> list[RetrievedMemory]:
        gathered: list[tuple[str, MemoryRecord, float]] = []
        for fname, mem in self.fields.items():
            if not mem.records:
                continue
            for rec, score in mem.retrieve(query, k=max(k, 10)):
                gathered.append((fname, rec, float(score)))
        best_by_text: dict[str, tuple[str, MemoryRecord, float]] = {}
        for fname, rec, score in gathered:
            prev = best_by_text.get(rec.text)
            if prev is None or score < prev[2]:
                best_by_text[rec.text] = (fname, rec, score)
        ordered = sorted(best_by_text.values(), key=lambda x: x[2])[: max(0, k)]
        return [RetrievedMemory(field_name=f, record=r, score=s, rank=i + 1) for i, (f, r, s) in enumerate(ordered)]
