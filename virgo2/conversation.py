from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .lifecycle import FieldLifecycleManager
from .session import SessionOverlay


@dataclass
class ConversationTurn:
    role: str
    text: str
    timestamp: str


class ConversationMemory:
    def __init__(self, lifecycle: FieldLifecycleManager) -> None:
        self.lifecycle = lifecycle

    def add_turn(self, role: str, text: str) -> ConversationTurn:
        ts = datetime.now(timezone.utc).isoformat()
        self.lifecycle.ingest(text, field_name="conversation_core", kind="conversation", metadata={"role": role, "timestamp": ts})
        return ConversationTurn(role=role, text=text, timestamp=ts)

    def process_user_message(self, text: str, session_id: str | None = None, auto_reflect: bool = True) -> dict[str, object]:
        if session_id:
            s = SessionOverlay(self.lifecycle, session_id=session_id)
            report = s.automate_after_turn("user", text)
            context = s.retrieve_context(text)
            return {"context": context, "stored_field": report["field_name"], "promoted_facts": report["promoted_facts"], "reflection_actions": report["reflection_actions"], "suggested_folds": []}
        ingest = self.lifecycle.ingest_auto(text, metadata={"role": "user"}, run_reflection=auto_reflect)
        context = self.context_for(text)
        reflection = ingest.get("reflection") or {}
        return {"context": context, "stored_field": ingest["field_name"], "promoted_facts": reflection.get("promoted_facts", []), "reflection_actions": reflection.get("actions", []), "suggested_folds": reflection.get("suggested_folds", [])}

    def context_for(self, user_message: str, k: int = 8) -> str:
        found = self.lifecycle.retrieve(user_message, k=k)
        return ("Relevant memory:\n"+"\n".join(f"- [{m.field_name} | {m.score:.3f}] {m.record.text}" for m in found)) if found else "No relevant memory found."

    def converse_memory_update(self, user_message: str, assistant_message: str | None = None) -> None:
        self.add_turn("user", user_message)
        if assistant_message:
            self.add_turn("assistant", assistant_message)
