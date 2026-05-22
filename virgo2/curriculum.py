from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CurriculumItem:
    text: str
    domain: str
    difficulty: int
    status: str = "pending"


class CurriculumQueue:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.items: list[CurriculumItem] = []
        self.load()

    def add(self, text: str, domain: str = "general", difficulty: int = 1) -> CurriculumItem:
        item = CurriculumItem(text=text, domain=domain, difficulty=int(difficulty))
        self.items.append(item)
        self.save()
        return item

    def next_batch(self, batch_size: int = 10) -> list[CurriculumItem]:
        return sorted([i for i in self.items if i.status == "pending"], key=lambda x: x.difficulty)[:batch_size]

    def mark_trained(self, items: list[CurriculumItem]) -> None:
        refs = {(i.text, i.domain, i.difficulty) for i in items}
        for item in self.items:
            if (item.text, item.domain, item.difficulty) in refs:
                item.status = "trained"
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["text\tdomain\tdifficulty\tstatus"]
        lines.extend(f"{i.text}\t{i.domain}\t{i.difficulty}\t{i.status}" for i in self.items)
        self.path.write_text("\n".join(lines), encoding="utf-8")

    def load(self) -> None:
        self.items = []
        if not self.path.exists():
            return
        lines = self.path.read_text(encoding="utf-8").splitlines()
        for line in lines[1:]:
            text, domain, difficulty, status = line.split("\t")
            self.items.append(CurriculumItem(text, domain, int(difficulty), status))
