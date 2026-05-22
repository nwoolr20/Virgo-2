from virgo2 import *  # noqa: F403


def test_public_imports_present():
    expected = [
        "CoordinateEncoder","NeuralField","MemoryRecord","NeuralMemory","ConversationMemory","ConversationTurn",
        "FieldLifecycleManager","FieldInfo","FieldRegistry","RetrievedMemory","MultiFieldRetriever","FieldVault",
        "FieldConsolidator","SemanticTaxonomy","ForgeLite","FieldType","ResolutionLevel","normalize_field_type",
        "normalize_resolution_level","FieldBuilder","FieldBuildRequest","FieldBuildResult","SessionOverlay","SessionInfo",
        "Conflict","ConflictPolicy","FieldConflictResolver","CurriculumItem","CurriculumQueue","ReflectionEngine","ReflectionReport",
    ]
    for name in expected:
        assert name in globals()
