from __future__ import annotations

from pathlib import Path
from .storage import save_memory
from .memory import NeuralMemory


def export_browser_bundle(memory: NeuralMemory, path: str | Path) -> None:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    save_memory(memory, out)
    (out / "manifest.txt").write_text(
        "bundle=virgo2-browser\nformat=field.npz+records.tsv\nversion=0.2.0\n",
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        "# Virgo-2 Browser Bundle\n\n"
        "This bundle contains `field.npz` and `records.tsv` for loading Virgo-2 memory in a future JS/WebGPU runtime.\n"
        "A loader should read coordinates and field parameters from `field.npz`, then associate text rows from `records.tsv`.\n",
        encoding="utf-8",
    )
