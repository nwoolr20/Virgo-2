from __future__ import annotations


class FieldType:
    TEXT = "text"
    CONVERSATION = "conversation"
    IDENTITY = "identity"
    PROJECT = "project"
    PROCEDURAL = "procedural"
    CURRICULUM = "curriculum"
    SUMMARY = "summary"
    FOLDED = "folded"
    DDIF_DISTILLED = "ddif_distilled"
    SESSION_OVERLAY = "session_overlay"


class ResolutionLevel:
    RAW = "raw"
    SESSION = "session"
    FOLDED = "folded"
    CORE = "core"


def normalize_field_type(value: str | None) -> str:
    valid = {v for k, v in FieldType.__dict__.items() if k.isupper()}
    candidate = (value or "").strip().lower()
    return candidate if candidate in valid else FieldType.TEXT


def normalize_resolution_level(value: str | None) -> str:
    valid = {v for k, v in ResolutionLevel.__dict__.items() if k.isupper()}
    candidate = (value or "").strip().lower()
    return candidate if candidate in valid else ResolutionLevel.RAW
