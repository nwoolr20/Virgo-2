from virgo2.taxonomy import SemanticTaxonomy


def test_taxonomy_routes():
    t = SemanticTaxonomy()
    assert t.field_for("my name is Nicholas") == "identity_core"
    assert t.field_for("Virgo-2 neural field project") == "project_core"
    assert t.field_for("how to train the model") == "procedural_core"
    assert t.field_for("hello there") == "conversation_core"


def test_taxonomy_confidence_stable():
    t = SemanticTaxonomy()
    d1 = t.classify("my name is Alex and I like coding")
    d2 = t.classify("my name is Alex and I like coding")
    assert d1.field_type == d2.field_type
    assert d1.confidence == d2.confidence
