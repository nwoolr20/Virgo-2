"""Virgo-2 memory substrate plus experimental DDiF-inspired neural-field LM."""

from .conflict import Conflict, ConflictPolicy, FieldConflictResolver
from .consolidation import FieldConsolidator
from .conversation import ConversationMemory, ConversationTurn
from .coordinates import CoordinateEncoder
from .curriculum import CurriculumItem, CurriculumQueue
from .field import NeuralField
from .field_builder import FieldBuildRequest, FieldBuildResult, FieldBuilder
from .field_types import FieldType, ResolutionLevel, normalize_field_type, normalize_resolution_level
from .forge import ForgeLite
from .lifecycle import FieldLifecycleManager
from .memory import MemoryRecord, NeuralMemory
from .reflection import ReflectionEngine, ReflectionReport
from .registry import FieldInfo, FieldRegistry
from .retrieval import MultiFieldRetriever, RetrievedMemory
from .session import SessionInfo, SessionOverlay
from .taxonomy import SemanticTaxonomy
from .vault import FieldVault

__all__ = [
    "Conflict",
    "ConflictPolicy",
    "FieldConflictResolver",
    "FieldConsolidator",
    "ConversationMemory",
    "ConversationTurn",
    "CoordinateEncoder",
    "CurriculumItem",
    "CurriculumQueue",
    "NeuralField",
    "FieldBuildRequest",
    "FieldBuildResult",
    "FieldBuilder",
    "FieldType",
    "ResolutionLevel",
    "normalize_field_type",
    "normalize_resolution_level",
    "ForgeLite",
    "FieldLifecycleManager",
    "MemoryRecord",
    "NeuralMemory",
    "ReflectionEngine",
    "ReflectionReport",
    "FieldInfo",
    "FieldRegistry",
    "MultiFieldRetriever",
    "RetrievedMemory",
    "SessionInfo",
    "SessionOverlay",
    "SemanticTaxonomy",
    "FieldVault",
]
