from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .lifecycle import FieldLifecycleManager


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
        salience = 1.0 if role == "user" else 0.5
        self.lifecycle.ingest(text, field_name="conversation_core", kind="conversation", metadata={"role": role, "timestamp": ts}, salience=salience)
        return ConversationTurn(role=role, text=text, timestamp=ts)

    def remember_fact(self, text: str, metadata: dict[str, str] | None = None, salience: float = 2.0) -> None:
        self.lifecycle.ingest(text, field_name=self.lifecycle.taxonomy.field_for(text), kind=self.lifecycle.taxonomy.kind_for(text), metadata=metadata, salience=salience)

    def context_for(self, user_message: str, k: int = 8) -> str:
        found = self.lifecycle.retrieve(user_message, k=k)
        if not found:
            return "No relevant memory found."
        lines = ["Relevant memory:"]
        for m in found:
            lines.append(f"- [{m.field_name} | {m.score:.3f}] {m.record.text}")
        return "\n".join(lines)

    def converse_memory_update(self, user_message: str, assistant_message: str | None = None) -> None:
        self.add_turn("user", user_message)
        t = user_message.lower()
        if any(k in t for k in ["remember", "my name is", "i like", "i prefer", "i am working on", "the project is", "my goal is"]):
            self.remember_fact(user_message, metadata={"source": "conversation"}, salience=2.5)
        if assistant_message:
            self.add_turn("assistant", assistant_message)
