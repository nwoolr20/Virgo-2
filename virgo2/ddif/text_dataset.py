from __future__ import annotations

from pathlib import Path


def read_texts(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
