from __future__ import annotations

from .consolidation import FieldConsolidator
from .field_types import FieldType
from .memory import MemoryRecord, NeuralMemory
from .reflection import ReflectionEngine
from .registry import FieldRegistry
from .retrieval import MultiFieldRetriever, RetrievedMemory
from .salience import estimate_salience, reinforce_salience
from .taxonomy import SemanticTaxonomy
from .vault import FieldVault

DEFAULT_FIELDS = {
    "conversation_core": FieldType.CONVERSATION,
    "text_core": FieldType.TEXT,
    "identity_core": FieldType.IDENTITY,
    "project_core": FieldType.PROJECT,
    "procedural_core": FieldType.PROCEDURAL,
}


class FieldLifecycleManager:
    def __init__(self, vault: FieldVault, registry: FieldRegistry | None = None, taxonomy: SemanticTaxonomy | None = None) -> None:
        self.vault = vault
        self.registry = registry or FieldRegistry(vault.root)
        self.registry.load()
        self.taxonomy = taxonomy or SemanticTaxonomy()
        self.fields: dict[str, NeuralMemory] = {}
        self.fit_pending: set[str] = set()
        self.maintenance_needed: set[str] = set()
        self.reflection_engine = ReflectionEngine(self)
        self.initialize_defaults()

    def initialize_defaults(self) -> None:
        for name, kind in DEFAULT_FIELDS.items():
            self.ensure_field(name, kind)
        self.registry.save()

    def ensure_field(self, name: str, kind: str = FieldType.TEXT) -> NeuralMemory:
        if name in self.fields:
            return self.fields[name]
        memory = self.vault.load(name) if self.vault.exists(name) else self.vault.create_empty(name)
        self.fields[name] = memory
        self.registry.register(name, kind, self.vault.field_path(name), field_type=kind)
        self.registry.update_count(name, len(memory.records))
        return memory

    def ingest(self, text: str, field_name: str | None = None, kind: str | None = None, metadata: dict[str, str] | None = None, salience: float | None = None) -> MemoryRecord:
        target = field_name or self.taxonomy.field_for(text)
        target_kind = kind or self.taxonomy.kind_for(text)
        memory = self.ensure_field(target, target_kind)
        self.registry.mark_dirty(target, True)
        record = memory.add(
            text,
            metadata=metadata,
            salience=salience if salience is not None else estimate_salience(text, (metadata or {}).get("role", "user")),
        )
        memory.fit()
        self.vault.save(target, memory)
        self.registry.update_count(target, len(memory.records))
        self.registry.mark_dirty(target, False)
        self.registry.save()
        return record

    def apply_pending_fits(self) -> list[str]:
        applied: list[str] = []
        for name in sorted(self.fit_pending):
            if name not in self.fields:
                continue
            mem = self.fields[name]
            mem.fit()
            self.vault.save(name, mem)
            self.registry.mark_dirty(name, False)
            applied.append(name)
        self.fit_pending.clear()
        self.registry.save()
        return applied

    def ingest_auto(self, text: str, metadata: dict[str, str] | None = None, salience: float | None = None, run_reflection: bool = True) -> dict[str, object]:
        kind = self.taxonomy.kind_for(text)
        field_name = self.taxonomy.field_for(text)
        record = self.ingest(text, field_name=field_name, kind=kind, metadata=metadata, salience=salience)

        reflection = None
        if run_reflection:
            report = self.reflection_engine.reflect_on_field(field_name, auto_promote=True, auto_fold=False)
            reflection = {
                "actions": report.actions_taken,
                "promoted_facts": report.promoted_facts,
                "suggested_folds": report.suggested_folds,
            }
        if len(self.fields[field_name].records) > 500:
            self.maintenance_needed.add(field_name)
        return {
            "field_name": field_name,
            "field_type": kind,
            "salience": record.salience,
            "record_count": len(self.fields[field_name].records),
            "reflection": reflection,
            "folded_fields": [],
        }

    def retrieve(self, query: str, k: int = 5, field_names: list[str] | None = None) -> list[RetrievedMemory]:
        names = field_names or [f.name for f in self.registry.list()]
        fields = {name: self.ensure_field(name, self.registry.get(name).kind if self.registry.get(name) else FieldType.TEXT) for name in names}
        results = MultiFieldRetriever(fields).search(query, k=k)
        for item in results:
            memory = self.fields[item.field_name]
            index = memory.records.index(item.record)
            memory.update_salience(index, reinforce_salience(memory.records[index].salience))
            self.fit_pending.add(item.field_name)
            self.registry.mark_dirty(item.field_name, True)
            self.registry.record_access(item.field_name)
        self.registry.save()
        return results

    def fold_if_needed(self, max_records_per_field: int = 500, target_prefix: str = "folded") -> list[str]:
        consolidator = FieldConsolidator(self.vault, self.registry)
        created: list[str] = []
        for info in self.registry.list():
            if info.record_count > max_records_per_field:
                target_name = f"{target_prefix}_{info.name}"
                consolidator.fold_field(info.name, target_name, max_records=max_records_per_field)
                created.append(target_name)
        self.registry.save()
        return created

    def maintenance_cycle(self, max_records_before_fold: int = 500, auto_fold: bool = True) -> dict[str, object]:
        fitted = self.apply_pending_fits()
        reports = self.reflection_engine.run_maintenance_cycle(max_records_before_fold, auto_fold)
        folded = self.fold_if_needed(max_records_before_fold) if auto_fold else []
        self.maintenance_needed.clear()
        return {
            "reflected_fields": [r.source_field for r in reports],
            "actions": [a for r in reports for a in r.actions_taken],
            "fitted_fields": fitted,
            "folded_fields": folded,
        }

    def save_all(self) -> None:
        for name, memory in self.fields.items():
            self.vault.save(name, memory)
            self.registry.update_count(name, len(memory.records))
            self.registry.mark_dirty(name, False)
        self.registry.save()

    def load_all(self) -> None:
        self.registry.load()
        for info in self.registry.list():
            if self.vault.exists(info.name):
                self.fields[info.name] = self.vault.load(info.name)

    def status(self) -> dict[str, object]:
        infos = self.registry.list()
        return {
            "vault_path": str(self.vault.root),
            "registry_path": str(self.registry.path),
            "registered_fields": [f.name for f in infos],
            "field_count": len(infos),
            "loaded_fields": sorted(self.fields.keys()),
            "record_counts": {f.name: f.record_count for f in infos},
            "dirty_fields": [f.name for f in infos if f.dirty],
            "fit_pending": sorted(self.fit_pending),
            "maintenance_needed": sorted(self.maintenance_needed),
        }

    def release_check(self) -> dict[str, object]:
        self.registry.load()
        infos = self.registry.list()
        dirty_fields = [f.name for f in infos if f.dirty]
        oversized = [f.name for f in infos if f.record_count > 500]
        lineage_invalid = [f.name for f in infos if f.kind == "folded" and not (f.parent_fields or f.folded_from)]
        ready = not dirty_fields and not self.fit_pending and not self.maintenance_needed and not lineage_invalid
        return {
            "registry_loaded": True,
            "field_count": len(infos),
            "dirty_fields": dirty_fields,
            "fit_pending": sorted(self.fit_pending),
            "maintenance_needed": sorted(self.maintenance_needed),
            "oversized_fields": oversized,
            "fold_lineage_invalid": lineage_invalid,
            "ready": ready,
        }
