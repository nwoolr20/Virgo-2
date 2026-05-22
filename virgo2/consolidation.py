from __future__ import annotations

from .memory import MemoryRecord, NeuralMemory
from .registry import FieldRegistry
from .vault import FieldVault


class FieldConsolidator:
    def __init__(self, vault: FieldVault, registry: FieldRegistry) -> None:
        self.vault = vault
        self.registry = registry

    def normalize_text(self, text: str) -> str:
        return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())

    def token_overlap(self, a: str, b: str) -> float:
        ta = set(self.normalize_text(a).split())
        tb = set(self.normalize_text(b).split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def detect_duplicate_records(self, memory: NeuralMemory) -> list[str]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for rec in memory.records:
            key = self.normalize_text(rec.text)
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        return duplicates

    def detect_overlapping_records(self, memory: NeuralMemory, threshold: float = 0.92) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for i, left in enumerate(memory.records):
            for j, right in enumerate(memory.records[i + 1 :], start=i + 1):
                if self.token_overlap(left.text, right.text) >= threshold:
                    pairs.append((i, j))
        return pairs

    def preserve_high_salience_records(self, records: list[MemoryRecord], limit: int) -> list[MemoryRecord]:
        return sorted(records, key=lambda r: r.salience, reverse=True)[:limit]

    def cluster_records(self, records: list[MemoryRecord], similarity_threshold: float = 0.35) -> list[list[MemoryRecord]]:
        clusters: list[list[MemoryRecord]] = []
        for record in records:
            placed = False
            for cluster in clusters:
                avg = sum(self.token_overlap(record.text, c.text) for c in cluster) / len(cluster)
                if avg >= similarity_threshold:
                    cluster.append(record)
                    placed = True
                    break
            if not placed:
                clusters.append([record])
        return clusters

    def create_summary_record(self, records: list[MemoryRecord]) -> MemoryRecord:
        texts = [r.text.strip() for r in records[:5]]
        summary = "; ".join(texts) if texts else "(empty)"
        salience = max(0.1, sum(r.salience for r in records) / max(1, len(records)))
        return MemoryRecord(
            text=f"Folded memory summary: {summary}",
            metadata={"folded": "true", "items": str(len(records))},
            salience=salience,
        )

    def fold_low_salience_records(self, records: list[MemoryRecord], max_summaries: int) -> list[MemoryRecord]:
        if not records or max_summaries <= 0:
            return []
        clusters = self.cluster_records(records)
        summaries = [self.create_summary_record(cluster) for cluster in clusters]
        return summaries[:max_summaries]

    def merge_fields(self, source_names: list[str], target_name: str, preserve_sources: bool = True) -> NeuralMemory:
        source_memories: dict[str, NeuralMemory] = {}
        for name in source_names:
            if not self.vault.exists(name):
                raise FileNotFoundError(f"source field not found: {name}")
            source_memories[name] = self.vault.load(name)

        merged = NeuralMemory()
        for name in source_names:
            for rec in source_memories[name].records:
                merged.add(rec.text, metadata=rec.metadata, salience=rec.salience)

        if merged.records:
            merged.fit()
        self.vault.save(target_name, merged)

        input_count = sum(len(source_memories[n].records) for n in source_names)
        compression_ratio = input_count / max(1, len(merged.records))

        self.registry.register(target_name, "folded", self.vault.field_path(target_name), field_type="folded")
        self.registry.update_count(target_name, len(merged.records))
        self.registry.set_folded_from(target_name, source_names)
        self.registry.set_parent_fields(target_name, source_names)
        self.registry.update_compression_ratio(target_name, compression_ratio)
        self.registry.mark_dirty(target_name, False)
        self.registry.save()
        return merged

    def fold_field(self, source_name: str, target_name: str, max_records: int = 250) -> NeuralMemory:
        if not self.vault.exists(source_name):
            raise FileNotFoundError(f"source field not found: {source_name}")
        source = self.vault.load(source_name)
        high_limit = max(1, max_records // 2)
        high = self.preserve_high_salience_records(source.records, high_limit)
        low = [r for r in source.records if r not in high]

        folded = NeuralMemory()
        for rec in high:
            folded.add(rec.text, rec.metadata, rec.salience)

        summaries = self.fold_low_salience_records(low, max_records - len(high))
        for rec in summaries:
            folded.add(rec.text, rec.metadata, rec.salience)

        if folded.records:
            folded.fit()
        self.vault.save(target_name, folded)

        compression_ratio = len(source.records) / max(1, len(folded.records))
        self.registry.register(target_name, "folded", self.vault.field_path(target_name), field_type="folded")
        self.registry.update_count(target_name, len(folded.records))
        self.registry.set_folded_from(target_name, [source_name])
        self.registry.set_parent_fields(target_name, [source_name])
        self.registry.update_compression_ratio(target_name, compression_ratio)
        self.registry.mark_dirty(target_name, False)
        self.registry.save()
        return folded
