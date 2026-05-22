from __future__ import annotations

from .memory import MemoryRecord, NeuralMemory
from .registry import FieldRegistry
from .vault import FieldVault


class FieldConsolidator:
    def __init__(self, vault: FieldVault, registry: FieldRegistry) -> None:
        self.vault = vault
        self.registry = registry

    def merge_fields(self, source_names: list[str], target_name: str, preserve_sources: bool = True) -> NeuralMemory:
        merged = NeuralMemory()
        for name in source_names:
            mem = self.vault.load(name)
            for rec in mem.records:
                merged.add(rec.text, metadata=rec.metadata, salience=rec.salience)
        if merged.records:
            merged.fit()
        self.vault.save(target_name, merged)
        info = self.registry.register(target_name, "folded", self.vault.field_path(target_name))
        self.registry.update_count(target_name, len(merged.records))
        self.registry.set_folded_from(info.name, source_names)
        self.registry.mark_dirty(target_name, False)
        if not preserve_sources:
            pass
        return merged

    def fold_field(self, source_name: str, target_name: str, max_records: int = 250) -> NeuralMemory:
        source = self.vault.load(source_name)
        if len(source.records) <= max_records:
            return self.merge_fields([source_name], target_name)
        ranked = sorted(source.records, key=lambda r: r.salience, reverse=True)
        keep_n = max(1, max_records // 2)
        kept = ranked[:keep_n]
        rest = ranked[keep_n:]
        target = NeuralMemory()
        for rec in kept:
            target.add(rec.text, rec.metadata, rec.salience)
        chunk_size = max(3, len(rest) // max(1, max_records - keep_n))
        for i in range(0, len(rest), chunk_size):
            target.add(**self.summarize_record_cluster(rest[i : i + chunk_size]).__dict__)
        if target.records:
            target.fit()
        self.vault.save(target_name, target)
        self.registry.register(target_name, "folded", self.vault.field_path(target_name))
        self.registry.update_count(target_name, len(target.records))
        self.registry.set_folded_from(target_name, [source_name])
        return target

    def summarize_record_cluster(self, records: list[MemoryRecord]) -> MemoryRecord:
        if not records:
            return MemoryRecord(text="Folded memory summary: (empty)", metadata={"folded": "true"}, salience=0.5)
        text = "; ".join(r.text.strip() for r in records[:5])
        salience = min(3.0, max(0.1, sum(r.salience for r in records) / len(records)))
        return MemoryRecord(
            text=f"Folded memory summary: {text}",
            metadata={"folded": "true", "items": str(len(records))},
            salience=salience,
        )
