from __future__ import annotations

import re
from dataclasses import dataclass

from .memory import MemoryRecord


@dataclass
class Conflict:
    key: str
    records: list[MemoryRecord]
    reason: str


class ConflictPolicy:
    TEMPORAL_NEWER_WINS = "temporal_newer_wins"
    HIGHER_SALIENCE_WINS = "higher_salience_wins"
    PRESERVE_BOTH = "preserve_both"
    FOLD_INTO_SUMMARY = "fold_into_summary"


class FieldConflictResolver:
    def _norm(self, t: str) -> str:
        return re.sub(r"\W+", " ", t.lower()).strip()

    def detect_conflicts(self, records: list[MemoryRecord]) -> list[Conflict]:
        grouped: dict[str, list[MemoryRecord]] = {}
        for r in records:
            grouped.setdefault(self._norm(r.text), []).append(r)
        out: list[Conflict] = []
        for k, recs in grouped.items():
            if len(recs) > 1:
                out.append(Conflict(k, recs, "normalized_duplicate"))
        return out

    def resolve(self, records: list[MemoryRecord], policy: str = ConflictPolicy.HIGHER_SALIENCE_WINS) -> list[MemoryRecord]:
        resolved: list[MemoryRecord] = []
        for c in self.detect_conflicts(records):
            if policy == ConflictPolicy.PRESERVE_BOTH:
                resolved.extend(c.records)
            else:
                resolved.append(max(c.records, key=lambda r: r.salience))
        normalized_conflicts = {id(r) for c in self.detect_conflicts(records) for r in c.records}
        resolved.extend([r for r in records if id(r) not in normalized_conflicts])
        return resolved
