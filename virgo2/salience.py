"""Salience helpers for fluid Virgo-2 memory."""

from __future__ import annotations

from .memory import NeuralMemory

MAX_SALIENCE = 5.0


def clamp_salience(value: float) -> float:
    return max(0.0, min(MAX_SALIENCE, float(value)))


def calculate_salience(
    text: str,
    role: str | None = None,
    explicit_importance: float | None = None,
) -> float:
    """Estimate memory importance using lightweight deterministic heuristics."""
    if explicit_importance is not None:
        return clamp_salience(explicit_importance)

    lowered = text.lower()
    score = 1.0

    if role == "assistant":
        score -= 0.25
    if any(phrase in lowered for phrase in ("remember", "my name is", "i am", "i'm", "i like", "i want")):
        score += 1.25
    if any(word in lowered for word in ("project", "goal", "virgo", "neural field", "memory", "research")):
        score += 0.75
    if any(word in lowered for word in ("important", "critical", "must", "always", "never")):
        score += 0.75
    if len(text) < 12:
        score -= 0.25

    return clamp_salience(score)


def decay_salience(memory: NeuralMemory, rate: float = 0.01) -> None:
    memory.decay(rate)


def reinforce_salience(memory: NeuralMemory, matched_indices: list[int], amount: float = 0.1) -> None:
    for index in matched_indices:
        if 0 <= index < len(memory.records):
            memory.records[index].salience = clamp_salience(memory.records[index].salience + amount)
