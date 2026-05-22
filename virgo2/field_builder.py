from __future__ import annotations

from dataclasses import dataclass

from .field_types import FieldType, normalize_field_type
from .memory import NeuralMemory
from .registry import FieldRegistry
from .salience import estimate_salience
from .taxonomy import SemanticTaxonomy
from .vault import FieldVault


@dataclass
class FieldBuildRequest:
    name: str
    text_items: list[str]
    field_type: str | None = None
    metadata: dict[str, str] | None = None
    salience: float | None = None
    auto_validate: bool = True


@dataclass
class FieldBuildResult:
    name: str
    record_count: int
    path: str
    field_type: str
    validation: dict[str, object]


class FieldBuilder:
    def __init__(self, vault: FieldVault, registry: FieldRegistry, taxonomy: SemanticTaxonomy | None = None) -> None:
        self.vault = vault
        self.registry = registry
        self.taxonomy = taxonomy or SemanticTaxonomy()

    def infer_field_type(self, text_items: list[str]) -> str:
        if not text_items:
            return FieldType.TEXT
        return normalize_field_type(self.taxonomy.kind_for(" ".join(text_items)))

    def validate_field(self, name: str, probe_queries: list[str] | None = None) -> dict[str, object]:
        memory = self.vault.load(name)
        probes = probe_queries or (memory.records[0].text.split()[:2] if memory.records else ["memory"])
        hits = sum(1 for query in probes if memory.retrieve(query, k=1))
        return {"probes": probes, "hits": hits, "ok": hits > 0 or not memory.records}

    def create_field(self, request: FieldBuildRequest) -> FieldBuildResult:
        field_type = normalize_field_type(request.field_type) if request.field_type else self.infer_field_type(request.text_items)
        memory = NeuralMemory()

        for text in request.text_items:
            if not text.strip():
                continue
            if request.salience is not None:
                salience = request.salience
            else:
                salience = estimate_salience(text, (request.metadata or {}).get("role", "user"))
            memory.add(text.strip(), metadata=request.metadata, salience=salience)

        if memory.records:
            memory.fit()

        self.vault.save(request.name, memory)
        self.registry.register(request.name, field_type, self.vault.field_path(request.name), field_type=field_type)
        self.registry.update_count(request.name, len(memory.records))
        self.registry.save()

        validation = self.validate_field(request.name) if request.auto_validate else {"ok": True}
        return FieldBuildResult(
            name=request.name,
            record_count=len(memory.records),
            path=str(self.vault.field_path(request.name)),
            field_type=field_type,
            validation=validation,
        )
