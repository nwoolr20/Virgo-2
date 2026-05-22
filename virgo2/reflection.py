from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .conflict import FieldConflictResolver
from .memory import MemoryRecord


@dataclass
class ReflectionReport:
    source_field: str
    repeated_themes: list[str]
    promoted_facts: list[str]
    suggested_folds: list[str]
    conflicts: list[str]
    actions_taken: list[str]


class ReflectionEngine:
    STOP = {"the", "a", "an", "to", "is", "are", "i", "my", "on", "and", "or", "of"}

    def __init__(self, lifecycle, conflict_resolver: FieldConflictResolver | None = None) -> None:
        self.lifecycle = lifecycle
        self.conflict_resolver = conflict_resolver or FieldConflictResolver()

    def extract_repeated_themes(self, records: list[MemoryRecord]) -> list[str]:
        tokens: list[str] = []
        for record in records:
            normalized = re.sub(r"[^a-z0-9\s]", " ", record.text.lower())
            for token in normalized.split():
                if token not in self.STOP and len(token) > 2:
                    tokens.append(token)
        counts = Counter(tokens)
        return [word for word, count in counts.most_common(8) if count > 1]

    def extract_stable_facts(self, records: list[MemoryRecord]) -> list[str]:
        keys = ["remember that", "my name is", "i am", "i'm", "i like", "i prefer", "my goal is", "the project is", "i am working on"]
        return [record.text.strip() for record in records if any(key in record.text.lower() for key in keys)]

    def promote_stable_facts(self, facts: list[str], source_field: str) -> list[str]:
        for fact in facts:
            self.lifecycle.ingest_auto(fact, metadata={"source_field": source_field}, run_reflection=False)
        return facts

    def suggest_folds(self, max_records: int = 500) -> list[str]:
        return [field.name for field in self.lifecycle.registry.list() if field.record_count > max_records]

    def reflect_on_field(self, field_name: str, auto_promote: bool = True, auto_fold: bool = False, max_records_before_fold: int = 500) -> ReflectionReport:
        info = self.lifecycle.registry.get(field_name)
        kind = info.kind if info else "text"
        memory = self.lifecycle.ensure_field(field_name, kind)

        themes = self.extract_repeated_themes(memory.records)
        facts = self.extract_stable_facts(memory.records)
        conflicts = [c.reason for c in self.conflict_resolver.detect_conflicts(memory.records)]

        actions: list[str] = []
        promoted: list[str] = []
        if auto_promote and facts:
            promoted = self.promote_stable_facts(facts, field_name)
            actions.append("promoted_facts")

        folds = [field_name] if len(memory.records) > max_records_before_fold else []
        if auto_fold and folds:
            self.lifecycle.fold_if_needed(max_records_per_field=max_records_before_fold)
            actions.append("auto_fold")

        return ReflectionReport(field_name, themes, promoted, folds, conflicts, actions)

    def reflect_on_recent_session(self, session_field: str, auto_promote: bool = True, auto_fold_session: bool = True) -> ReflectionReport:
        return self.reflect_on_field(session_field, auto_promote=auto_promote, auto_fold=auto_fold_session, max_records_before_fold=50)

    def run_maintenance_cycle(self, max_records_before_fold: int = 500, auto_fold: bool = True) -> list[ReflectionReport]:
        reports: list[ReflectionReport] = []
        for field in self.lifecycle.registry.list():
            reports.append(
                self.reflect_on_field(
                    field.name,
                    auto_promote=True,
                    auto_fold=auto_fold,
                    max_records_before_fold=max_records_before_fold,
                )
            )
        return reports
