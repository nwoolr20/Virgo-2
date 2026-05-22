from __future__ import annotations

from pathlib import Path

from .registry import FieldRegistry
from .vault import FieldVault


class ForgeLite:
    def __init__(self, vault: FieldVault, registry: FieldRegistry) -> None:
        self.vault = vault
        self.registry = registry

    def run_checks(self) -> dict[str, object]:
        self.registry.load()
        report = self.vault.integrity_report()
        dirty = [f.name for f in self.registry.list() if f.dirty]
        oversized = [f.name for f in self.registry.list() if f.record_count > 500]
        missing_paths = [f.name for f in self.registry.list() if not Path(f.path).exists()]
        negative_salience: list[str] = []
        loadable_count = 0
        for f in self.registry.list():
            if self.vault.exists(f.name):
                mem = self.vault.load(f.name)
                loadable_count += 1
                if mem.records:
                    _ = mem.retrieve("check", k=1)
                for r in mem.records:
                    if r.salience < 0:
                        negative_salience.append(f.name)
                        break
        return {
            "registry_loaded": True,
            "vault_exists": self.vault.root.exists(),
            "loadable_fields_count": loadable_count,
            "dirty_fields": dirty,
            "oversized_fields": oversized,
            "missing_registered_paths": missing_paths,
            "missing_field_artifacts": report["missing_field_artifacts"],
            "errors": report["errors"],
            "negative_salience_fields": negative_salience,
        }

    def write_report(self, path: str | Path) -> None:
        checks = self.run_checks()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# ForgeLite Report", ""]
        for k, v in checks.items():
            lines.append(f"- {k}: {v}")
        p.write_text("\n".join(lines), encoding="utf-8")
