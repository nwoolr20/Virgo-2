from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SalienceConfig:
    min_salience: float = 0.1
    max_salience: float = 5.0
    reinforce_step: float = 0.15


def clamp_salience(value: float, min_salience: float = 0.1, max_salience: float = 5.0) -> float:
    return max(min_salience, min(max_salience, float(value)))


def estimate_salience(text: str, role: str = "user") -> float:
    t = text.lower()
    score = 1.0
    strong = ["remember", "my name is", "i prefer", "i like", "goal", "important"]
    project = ["project", "virgo", "neural field", "ddif", "architecture"]
    if any(k in t for k in strong):
        score += 1.0
    if any(k in t for k in project):
        score += 0.5
    if role == "assistant":
        score *= 0.6
    return clamp_salience(score)


def decay_salience(value: float, rate: float = 0.01) -> float:
    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1")
    return clamp_salience(value * (1.0 - rate))


def reinforce_salience(value: float, step: float = 0.15) -> float:
    return clamp_salience(value + step)
