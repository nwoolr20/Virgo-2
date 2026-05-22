from virgo2.taxonomy import SemanticTaxonomy


def test_taxonomy_routes():
    t = SemanticTaxonomy()
    assert t.field_for("my name is Nicholas") == "identity_core"
    assert t.field_for("Virgo-2 neural field project") == "project_core"
    assert t.field_for("how to train the model") == "procedural_core"
    assert t.field_for("hello there") in {"conversation_core", "semantic_core"}
