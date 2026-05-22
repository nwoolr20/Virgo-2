from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .field_types import ResolutionLevel, normalize_resolution_level


REGISTRY_VERSION = "2"
LEGACY_HEADER = ["name", "kind", "path", "record_count", "created_at", "updated_at", "dirty", "folded_from"]


@dataclass
class FieldInfo:
    name: str
    kind: str
    path: str
    record_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    dirty: bool = False
    folded_from: list[str] | None = None
    field_type: str = ""
    domain: str = ""
    description: str = ""
    source_count: int = 0
    compression_ratio: float = 1.0
    parent_fields: list[str] | None = None
    last_accessed_at: str = ""
    query_count: int = 0
    health: str = "unknown"
    resolution_level: str = ResolutionLevel.RAW


class FieldRegistry:
    HEADER = [
        "registry_version",
        "name",
        "kind",
        "path",
        "record_count",
        "created_at",
        "updated_at",
        "dirty",
        "folded_from",
        "field_type",
        "domain",
        "description",
        "source_count",
        "compression_ratio",
        "parent_fields",
        "last_accessed_at",
        "query_count",
        "health",
        "resolution_level",
    ]

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.path = self.root / "registry.tsv"
        self._fields: dict[str, FieldInfo] = {}

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _escape(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")

    def _unescape(self, value: str) -> str:
        out = value
        out = out.replace("\\n", "\n")
        out = out.replace("\\t", "\t")
        out = out.replace("\\\\", "\\")
        return out

    def register(self, name: str, kind: str, path: str | Path, **kwargs: object) -> FieldInfo:
        now = self._now()
        info = self._fields.get(name)
        if info is None:
            info = FieldInfo(name=name, kind=kind, path=str(path), created_at=now, updated_at=now, folded_from=[], parent_fields=[])
            self._fields[name] = info
        info.kind = kind
        info.path = str(path)
        info.updated_at = now
        for key, value in kwargs.items():
            if hasattr(info, key):
                setattr(info, key, value)
        return info

    def get(self, name: str) -> FieldInfo | None:
        return self._fields.get(name)

    def list(self, kind: str | None = None) -> list[FieldInfo]:
        values = list(self._fields.values())
        if kind is not None:
            values = [v for v in values if v.kind == kind]
        return sorted(values, key=lambda v: v.name)

    def mark_dirty(self, name: str, dirty: bool = True) -> None:
        info = self._fields[name]
        info.dirty = dirty
        info.updated_at = self._now()

    def update_count(self, name: str, count: int) -> None:
        info = self._fields[name]
        info.record_count = int(count)
        info.updated_at = self._now()

    def set_folded_from(self, name: str, sources: list[str]) -> None:
        info = self._fields[name]
        info.folded_from = list(sources)
        info.updated_at = self._now()

    def record_access(self, name: str) -> None:
        info = self._fields[name]
        now = self._now()
        info.last_accessed_at = now
        info.query_count += 1
        info.updated_at = now

    def update_health(self, name: str, health: str) -> None:
        info = self._fields[name]
        info.health = health
        info.updated_at = self._now()

    def update_compression_ratio(self, name: str, ratio: float) -> None:
        info = self._fields[name]
        info.compression_ratio = float(ratio)
        info.updated_at = self._now()

    def set_parent_fields(self, name: str, parents: list[str]) -> None:
        info = self._fields[name]
        info.parent_fields = list(parents)
        info.updated_at = self._now()

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            fh.write("\t".join(self.HEADER) + "\n")
            for f in self.list():
                row = [
                    REGISTRY_VERSION,
                    f.name,
                    f.kind,
                    f.path,
                    str(f.record_count),
                    f.created_at,
                    f.updated_at,
                    "1" if f.dirty else "0",
                    ",".join(f.folded_from or []),
                    f.field_type,
                    f.domain,
                    f.description,
                    str(f.source_count),
                    str(f.compression_ratio),
                    ",".join(f.parent_fields or []),
                    f.last_accessed_at,
                    str(f.query_count),
                    f.health,
                    normalize_resolution_level(f.resolution_level),
                ]
                fh.write("\t".join(self._escape(v) for v in row) + "\n")

    def _from_data(self, data: dict[str, str]) -> FieldInfo:
        return FieldInfo(
            name=data.get("name", ""),
            kind=data.get("kind", "semantic"),
            path=data.get("path", ""),
            record_count=int(data.get("record_count", "0") or 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            dirty=data.get("dirty", "0") == "1",
            folded_from=[x for x in data.get("folded_from", "").split(",") if x],
            field_type=data.get("field_type", ""),
            domain=data.get("domain", ""),
            description=data.get("description", ""),
            source_count=int(data.get("source_count", "0") or 0),
            compression_ratio=float(data.get("compression_ratio", "1") or 1),
            parent_fields=[x for x in data.get("parent_fields", "").split(",") if x],
            last_accessed_at=data.get("last_accessed_at", ""),
            query_count=int(data.get("query_count", "0") or 0),
            health=data.get("health", "unknown"),
            resolution_level=normalize_resolution_level(data.get("resolution_level")),
        )

    def load(self) -> None:
        self._fields = {}
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            header = [self._unescape(x) for x in next(fh, "\n").rstrip("\n").split("\t")]
            for line in fh:
                cols = [self._unescape(x) for x in line.rstrip("\n").split("\t")]
                if len(cols) != len(header):
                    continue
                data = dict(zip(header, cols, strict=False))
                if header == LEGACY_HEADER:
                    data["registry_version"] = "1"
                name = data.get("name", "")
                if not name:
                    continue
                self._fields[name] = self._from_data(data)
