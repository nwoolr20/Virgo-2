from __future__ import annotations

from pathlib import Path

from .memory import NeuralMemory
from .storage import load_memory, save_memory


class FieldVault:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.fields_root = self.root / "fields"

    def field_path(self, name: str) -> Path:
        return self.fields_root / name

    def exists(self, name: str) -> bool:
        p = self.field_path(name)
        return (p / "field.npz").exists() and (p / "records.tsv").exists()

    def create_empty(self, name: str) -> NeuralMemory:
        _ = self.field_path(name)
        return NeuralMemory()

    def save(self, name: str, memory: NeuralMemory) -> None:
        save_memory(memory, self.field_path(name))

    def load(self, name: str) -> NeuralMemory:
        p = self.field_path(name)
        if not self.exists(name):
            raise FileNotFoundError(f"Field '{name}' not found at {p}")
        return load_memory(p)

    def list_field_names(self) -> list[str]:
        if not self.fields_root.exists():
            return []
        names = [p.name for p in self.fields_root.iterdir() if p.is_dir()]
        return sorted(names)

    def integrity_report(self) -> dict[str, object]:
        missing: list[str] = []
        loadable: list[str] = []
        errors: list[str] = []
        names = self.list_field_names()
        for name in names:
            p = self.field_path(name)
            if not (p / "field.npz").exists() or not (p / "records.tsv").exists():
                missing.append(name)
                continue
            try:
                _ = self.load(name)
                loadable.append(name)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")
        return {
            "vault_path": str(self.root),
            "field_count": len(names),
            "missing_field_artifacts": missing,
            "loadable_fields": loadable,
            "errors": errors,
        }
