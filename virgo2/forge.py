from __future__ import annotations

import time
from pathlib import Path

from .registry import FieldRegistry
from .vault import FieldVault


class ForgeLite:
    def __init__(self, vault: FieldVault, registry: FieldRegistry) -> None:
        self.vault = vault
        self.registry = registry

    def validate_field_recall(self, field_name: str | None = None) -> dict[str, object]:
        names = [field_name] if field_name else [f.name for f in self.registry.list()]
        results: dict[str, object] = {}
        for name in names:
            if not self.vault.exists(name):
                continue
            memory = self.vault.load(name)
            probes = memory.records[: min(3, len(memory.records))]
            hits = 0
            for rec in probes:
                words = rec.text.split()
                query = " ".join(words[: min(4, len(words))])
                returned = memory.retrieve(query, k=3)
                if returned and any(r.text == rec.text for r, _ in returned):
                    hits += 1
            score = hits / max(1, len(probes))
            results[name] = {"probes": len(probes), "hits": hits, "score": score, "ok": score >= 0.5 if probes else True}
        return {"recall": results}

    def validate_field_roundtrip(self, field_name: str | None = None) -> dict[str, object]:
        names = [field_name] if field_name else [f.name for f in self.registry.list()]
        checks: dict[str, bool] = {}
        for name in names:
            if not self.vault.exists(name):
                continue
            before = self.vault.load(name)
            self.vault.save(name, before)
            after = self.vault.load(name)
            same_len = len(before.records) == len(after.records)
            same_records = all(
                b.text == a.text and b.salience == a.salience and (b.metadata or {}) == (a.metadata or {})
                for b, a in zip(before.records, after.records)
            ) if same_len else False
            checks[name] = same_len and same_records
        return {"roundtrip": checks}

    def validate_conversation_recall(self) -> dict[str, object]:
        return self.validate_field_recall("conversation_core")

    def validate_folding_lineage(self) -> dict[str, object]:
        invalid = [f.name for f in self.registry.list() if f.kind == "folded" and not (f.parent_fields or f.folded_from)]
        return {"lineage_valid": not invalid, "invalid": invalid}

    def benchmark_query_latency(self, query: str = "memory") -> dict[str, object]:
        load_ms = 0.0
        retrieve_ms = 0.0
        total_start = time.perf_counter()
        for f in self.registry.list():
            if not self.vault.exists(f.name):
                continue
            s = time.perf_counter()
            mem = self.vault.load(f.name)
            load_ms += (time.perf_counter() - s) * 1000
            s = time.perf_counter()
            mem.retrieve(query, k=1)
            retrieve_ms += (time.perf_counter() - s) * 1000
        total_ms = (time.perf_counter() - total_start) * 1000
        return {"query": query, "load_latency_ms": load_ms, "retrieval_latency_ms": retrieve_ms, "total_latency_ms": total_ms}

    def run_checks(self) -> dict[str, object]:
        self.registry.load()
        report = self.vault.integrity_report()
        checks = {
            "field_count": len(self.registry.list()),
            "loadable_fields": report["loadable_fields"],
            "loadable_fields_count": len(report["loadable_fields"]),
            "empty_fields": [f.name for f in self.registry.list() if f.record_count == 0],
            "dirty_fields": [f.name for f in self.registry.list() if f.dirty],
            "oversized_fields": [f.name for f in self.registry.list() if f.record_count > 500],
            "lineage": self.validate_folding_lineage(),
            "recall": self.validate_field_recall(),
            "roundtrip": self.validate_field_roundtrip(),
            "latency": self.benchmark_query_latency(),
            "missing_registered_paths": [f.name for f in self.registry.list() if not Path(f.path).exists()],
            "errors": report["errors"],
        }
        checks["recommended_maintenance_actions"] = ["maintenance-cycle"] if checks["oversized_fields"] else ["none"]
        return checks

    def write_report(self, path: str | Path) -> None:
        checks = self.run_checks()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# ForgeLite Report", ""]
        for key, value in checks.items():
            lines.append(f"- {key}: {value}")
        p.write_text("\n".join(lines), encoding="utf-8")
