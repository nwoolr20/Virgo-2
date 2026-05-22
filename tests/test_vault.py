import pytest

from virgo2.memory import NeuralMemory
from virgo2.vault import FieldVault


def test_vault_ops(tmp_path) -> None:
    v = FieldVault(tmp_path)
    assert v.field_path("conversation_core") == tmp_path / "fields" / "conversation_core"

    m = NeuralMemory()
    m.add("hello")
    m.fit()
    v.save("conversation_core", m)

    assert "conversation_core" in v.list_field_names()
    loaded = v.load("conversation_core")
    assert loaded.records[0].text == "hello"
    rep = v.integrity_report()
    assert {"vault_path", "field_count", "missing_field_artifacts", "loadable_fields", "errors"} <= set(rep)
    with pytest.raises(FileNotFoundError):
        v.load("missing")
