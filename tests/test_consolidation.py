from virgo2.consolidation import FieldConsolidator
from virgo2.memory import NeuralMemory
from virgo2.registry import FieldRegistry
from virgo2.vault import FieldVault


def test_consolidation(tmp_path) -> None:
    v = FieldVault(tmp_path)
    r = FieldRegistry(tmp_path)
    m = NeuralMemory()
    for i in range(20):
        m.add(f"item {i}", salience=3.0 if i < 2 else 0.5)
    m.fit()
    v.save("conversation_core", m)
    r.register("conversation_core", "conversation", v.field_path("conversation_core"))
    r.update_count("conversation_core", len(m.records))

    c = FieldConsolidator(v, r)
    merged = c.merge_fields(["conversation_core"], "merged")
    assert len(merged.records) == 20
    folded = c.fold_field("conversation_core", "folded", max_records=8)
    assert len(folded.records) <= 9
    assert any("item 0" in rec.text for rec in folded.records)
    assert r.get("folded").folded_from == ["conversation_core"]
