from __future__ import annotations

from dataclasses import dataclass

from .field_types import FieldType


@dataclass(frozen=True)
class TaxonomyResult:
    field_name: str
    field_type: str
    tags: list[str]
    confidence: float
    scores: dict[str, float]


class SemanticTaxonomy:
    PRIORITY = [FieldType.IDENTITY, FieldType.PROJECT, FieldType.PROCEDURAL, FieldType.TEXT, FieldType.CONVERSATION]
    KEYWORDS: dict[str, dict[str, float]] = {
        FieldType.IDENTITY: {"my name is": 3.0, "i am": 1.3, "i'm": 1.3, "i like": 1.2, "i prefer": 1.2},
        FieldType.PROJECT: {"project": 2.0, "architecture": 1.5, "repository": 1.2, "code": 1.1, "neural field": 1.8, "virgo": 1.8},
        FieldType.PROCEDURAL: {"how to": 2.6, "steps": 1.4, "workflow": 1.5, "procedure": 1.5},
        FieldType.TEXT: {" is ": 0.8, " are ": 0.8, " means ": 0.9, "defined": 0.7},
        FieldType.CONVERSATION: {"hello": 0.8, "thanks": 0.8, "hi": 0.7},
    }

    def score_text(self, text: str) -> dict[str, float]:
        lower = f" {text.lower()} "
        scores = {k: 0.0 for k in self.PRIORITY}
        for field_type in self.PRIORITY:
            for phrase, weight in self.KEYWORDS[field_type].items():
                if phrase in lower:
                    scores[field_type] += weight
        if max(scores.values()) == 0.0:
            scores[FieldType.CONVERSATION] = 0.1
        return scores

    def classify(self, text: str) -> TaxonomyResult:
        scores = self.score_text(text)
        ranked = sorted(self.PRIORITY, key=lambda item: (-scores[item], self.PRIORITY.index(item)))
        selected = ranked[0]
        total = sum(scores.values())
        confidence = scores[selected] / total if total > 0 else 0.0
        confidence = min(1.0, max(0.0, confidence))
        tags = [item for item in ranked if scores[item] > 0]
        field_name = f"{selected}_core"
        return TaxonomyResult(field_name=field_name, field_type=selected, tags=tags, confidence=confidence, scores=scores)

    def tags_for(self, text: str) -> list[str]:
        return self.classify(text).tags

    def field_for(self, text: str) -> str:
        return self.classify(text).field_name

    def kind_for(self, text: str) -> str:
        return self.classify(text).field_type
