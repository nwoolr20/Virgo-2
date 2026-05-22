from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from .field_types import FieldType
from .reflection import ReflectionEngine


@dataclass
class SessionInfo:
    session_id: str
    field_name: str
    promoted_count: int = 0
    folded: bool = False


class SessionOverlay:
    def __init__(self, lifecycle, session_id: str | None = None) -> None:
        sid = session_id or uuid4().hex[:8]
        self.lifecycle = lifecycle
        self.info = SessionInfo(session_id=sid, field_name=f"session_{sid}")
        self.lifecycle.ensure_field(self.info.field_name, FieldType.SESSION_OVERLAY)

    def add_turn(self, role: str, text: str) -> None:
        self.lifecycle.ingest(text, field_name=self.info.field_name, kind=FieldType.SESSION_OVERLAY, metadata={"role": role, "session_id": self.info.session_id})

    def retrieve_context(self, query: str, k: int = 8) -> str:
        hits = self.lifecycle.retrieve(query, k=k, field_names=[self.info.field_name] + [f.name for f in self.lifecycle.registry.list() if not f.name.startswith("session_")])
        return "\n".join(h.record.text for h in hits)

    def promote_stable_facts(self) -> list[str]:
        rep = ReflectionEngine(self.lifecycle).reflect_on_recent_session(self.info.field_name, auto_promote=True, auto_fold_session=False)
        self.info.promoted_count += len(rep.promoted_facts)
        return rep.promoted_facts

    def fold_session(self, target_field: str | None = None) -> str:
        target = target_field or f"folded_{self.info.field_name}"
        self.lifecycle.fold_if_needed(max_records_per_field=1, target_prefix=f"folded_{self.info.session_id}")
        self.info.folded = True
        return target

    def automate_after_turn(self, role: str, text: str) -> dict[str, object]:
        self.add_turn(role, text)
        rep = ReflectionEngine(self.lifecycle).reflect_on_recent_session(self.info.field_name, auto_promote=True, auto_fold_session=False)
        return {"session_id": self.info.session_id, "field_name": self.info.field_name, "promoted_facts": rep.promoted_facts, "reflection_actions": rep.actions_taken}
