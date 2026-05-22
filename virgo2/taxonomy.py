from __future__ import annotations


class SemanticTaxonomy:
    def tags_for(self, text: str) -> list[str]:
        t = text.lower()
        tags: list[str] = []
        if any(k in t for k in ["my name is", "i am", "i'm", "i like", "i prefer", "remember that i"]):
            tags.append("identity")
        if any(k in t for k in ["project", "virgo", "neural field", "ddif", "repository", "github", "code", "architecture"]):
            tags.append("project")
        if any(k in t for k in ["how to", "steps", "workflow", "process", "procedure"]):
            tags.append("procedural")
        if not tags:
            tags.append("conversation")
        return tags

    def field_for(self, text: str) -> str:
        tags = self.tags_for(text)
        if "identity" in tags:
            return "identity_core"
        if "project" in tags:
            return "project_core"
        if "procedural" in tags:
            return "procedural_core"
        t = text.lower()
        if any(x in t for x in ["fact", "is", "are", "was", "were"]):
            return "semantic_core"
        return "conversation_core"

    def kind_for(self, text: str) -> str:
        return self.field_for(text).replace("_core", "")
