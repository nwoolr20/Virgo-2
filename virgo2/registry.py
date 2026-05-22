from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


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


class FieldRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.path = self.root / "registry.tsv"
        self._fields: dict[str, FieldInfo] = {}

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def register(self, name: str, kind: str, path: str | Path) -> FieldInfo:
        p = str(Path(path))
        now = self._now()
        if name in self._fields:
            f = self._fields[name]
            f.kind = kind
            f.path = p
            f.updated_at = now
            return f
        info = FieldInfo(name=name, kind=kind, path=p, created_at=now, updated_at=now, folded_from=[])
        self._fields[name] = info
        return info

    def get(self, name: str) -> FieldInfo | None:
        return self._fields.get(name)

    def list(self, kind: str | None = None) -> list[FieldInfo]:
        vals = list(self._fields.values())
        if kind is None:
            return sorted(vals, key=lambda x: x.name)
        return sorted([f for f in vals if f.kind == kind], key=lambda x: x.name)

    def mark_dirty(self, name: str, dirty: bool = True) -> None:
        f = self._fields[name]
        f.dirty = dirty
        f.updated_at = self._now()

    def update_count(self, name: str, count: int) -> None:
        f = self._fields[name]
        f.record_count = int(count)
        f.updated_at = self._now()

    def set_folded_from(self, name: str, sources: list[str]) -> None:
        f = self._fields[name]
        f.folded_from = list(sources)
        f.updated_at = self._now()

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            fh.write("name\tkind\tpath\trecord_count\tcreated_at\tupdated_at\tdirty\tfolded_from\n")
            for f in self.list():
                folded = ",".join(f.folded_from or [])
                fh.write(
                    f"{f.name}\t{f.kind}\t{f.path}\t{f.record_count}\t{f.created_at}\t{f.updated_at}\t{int(f.dirty)}\t{folded}\n"
                )

    def load(self) -> None:
        self._fields = {}
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            next(fh, None)
            for line in fh:
                name, kind, path, count, created, updated, dirty, folded = line.rstrip("\n").split("\t")
                self._fields[name] = FieldInfo(
                    name=name,
                    kind=kind,
                    path=path,
                    record_count=int(count),
                    created_at=created,
                    updated_at=updated,
                    dirty=dirty == "1",
                    folded_from=[x for x in folded.split(",") if x] if folded else [],
                )
