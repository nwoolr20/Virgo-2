from __future__ import annotations

from dataclasses import dataclass

from .field_types import FieldType


@dataclass(frozen=True)
class TaxonomyDecision:
    field_type: str
    confidence: float
    tags: list[str]


class SemanticTaxonomy:
    PRIORITY = [FieldType.IDENTITY, FieldType.PROJECT, FieldType.PROCEDURAL, FieldType.CONVERSATION]
    KEYWORDS: dict[str, dict[str, float]] = {
        FieldType.IDENTITY: {
            "my name is": 2.5,
            "i am": 1.2,
            "i'm": 1.2,
            "i like": 1.0,
            "i prefer": 1.0,
            "remember that i": 1.4,
        },
        FieldType.PROJECT: {
            "project": 2.0,
            "virgo": 1.5,
            "neural field": 1.5,
            "ddif": 1.4,
            "repository": 1.1,
            "architecture": 1.2,
            "code": 0.8,
        },
        FieldType.PROCEDURAL: {
            "how to": 2.0,
            "steps": 1.1,
            "workflow": 1.2,
            "process": 1.2,
            "procedure": 1.3,
        },
    }

    def score_text(self, text: str) -> dict[str, float]:
        lower = text.lower()
        scores = {k: 0.0 for k in self.PRIORITY}
        for tag, weighted_keywords in self.KEYWORDS.items():
            for phrase, weight in weighted_keywords.items():
                if phrase in lower:
                    scores[tag] += weight
        if max(scores.values()) == 0.0:
            scores[FieldType.CONVERSATION] = 0.1
        return scores

    def classify(self, text: str) -> TaxonomyDecision:
        scores = self.score_text(text)
        ranked = sorted(self.PRIORITY, key=lambda t: (-scores[t], self.PRIORITY.index(t)))
        top = ranked[0]
        total = sum(scores.values()) or 1.0
        confidence = scores[top] / total
        tags = [tag for tag in ranked if scores[tag] > 0]
        return TaxonomyDecision(field_type=top, confidence=confidence, tags=tags)

    def tags_for(self, text: str) -> list[str]:
        return self.classify(text).tags or [FieldType.CONVERSATION]

    def kind_for(self, text: str) -> str:
        return self.classify(text).field_type

    def field_for(self, text: str) -> str:
        return f"{self.kind_for(text)}_core"
