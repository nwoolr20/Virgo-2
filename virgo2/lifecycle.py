from __future__ import annotations

from .consolidation import FieldConsolidator
from .memory import MemoryRecord, NeuralMemory
from .registry import FieldRegistry
from .retrieval import MultiFieldRetriever, RetrievedMemory
from .salience import estimate_salience, reinforce_salience
from .taxonomy import SemanticTaxonomy
from .vault import FieldVault

DEFAULT_FIELDS = {
    "conversation_core": "conversation",
    "semantic_core": "semantic",
    "identity_core": "identity",
    "project_core": "project",
    "procedural_core": "procedural",
}


class FieldLifecycleManager:
    def __init__(self, vault: FieldVault, registry: FieldRegistry | None = None, taxonomy: SemanticTaxonomy | None = None) -> None:
        self.vault = vault
        self.registry = registry or FieldRegistry(vault.root)
        self.registry.load()
        self.taxonomy = taxonomy or SemanticTaxonomy()
        self.fields: dict[str, NeuralMemory] = {}
        self.initialize_defaults()

    def initialize_defaults(self) -> None:
        for name, kind in DEFAULT_FIELDS.items():
            self.ensure_field(name, kind)
        self.registry.save()

    def ensure_field(self, name: str, kind: str = "semantic") -> NeuralMemory:
        if name in self.fields:
            return self.fields[name]
        mem = self.vault.load(name) if self.vault.exists(name) else self.vault.create_empty(name)
        self.fields[name] = mem
        self.registry.register(name, kind, self.vault.field_path(name))
        self.registry.update_count(name, len(mem.records))
        return mem

    def ingest(self, text: str, field_name: str | None = None, kind: str | None = None, metadata: dict[str, str] | None = None, salience: float | None = None) -> MemoryRecord:
        fname = field_name or self.taxonomy.field_for(text)
        fkind = kind or self.taxonomy.kind_for(text)
        mem = self.ensure_field(fname, fkind)
        self.registry.mark_dirty(fname, True)
        rec = mem.add(text, metadata=metadata, salience=salience if salience is not None else estimate_salience(text, metadata.get("role", "user") if metadata else "user"))
        mem.fit()
        self.vault.save(fname, mem)
        self.registry.update_count(fname, len(mem.records))
        self.registry.mark_dirty(fname, False)
        self.registry.save()
        return rec

    def retrieve(self, query: str, k: int = 5, field_names: list[str] | None = None) -> list[RetrievedMemory]:
        names = field_names or [f.name for f in self.registry.list()]
        fields = {name: self.ensure_field(name, self.registry.get(name).kind if self.registry.get(name) else "semantic") for name in names}
        out = MultiFieldRetriever(fields).search(query, k=k)
        for item in out:
            mem = self.fields[item.field_name]
            idx = mem.records.index(item.record)
            mem.update_salience(idx, reinforce_salience(mem.records[idx].salience))
            mem.fit()
            self.vault.save(item.field_name, mem)
            self.registry.update_count(item.field_name, len(mem.records))
        self.registry.save()
        return out

    def save_all(self) -> None:
        for name, mem in self.fields.items():
            self.vault.save(name, mem)
            self.registry.update_count(name, len(mem.records))
            self.registry.mark_dirty(name, False)
        self.registry.save()

    def load_all(self) -> None:
        self.registry.load()
        for info in self.registry.list():
            if self.vault.exists(info.name):
                self.fields[info.name] = self.vault.load(info.name)

    def fold_if_needed(self, max_records_per_field: int = 500, target_prefix: str = "folded") -> list[str]:
        consolidator = FieldConsolidator(self.vault, self.registry)
        created: list[str] = []
        for info in self.registry.list():
            if info.record_count > max_records_per_field:
                target = f"{target_prefix}_{info.name}"
                consolidator.fold_field(info.name, target, max_records=max_records_per_field)
                created.append(target)
        self.registry.save()
        return created

    def status(self) -> dict[str, object]:
        return {
            "vault_path": str(self.vault.root),
            "registered_fields": [f.name for f in self.registry.list()],
            "loaded_fields": sorted(self.fields.keys()),
            "record_counts": {f.name: f.record_count for f in self.registry.list()},
            "dirty_fields": [f.name for f in self.registry.list() if f.dirty],
        }
