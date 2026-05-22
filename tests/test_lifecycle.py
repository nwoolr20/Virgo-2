from virgo2.lifecycle import FieldLifecycleManager
from virgo2.registry import FieldRegistry
from virgo2.vault import FieldVault


def test_lifecycle(tmp_path):
    m = FieldLifecycleManager(FieldVault(tmp_path), FieldRegistry(tmp_path))
    s = m.status()
    assert "conversation_core" in s["registered_fields"]
    m.ingest("my name is Nicholas")
    out = m.retrieve("Nicholas")
    assert out
    m2 = FieldLifecycleManager(FieldVault(tmp_path), FieldRegistry(tmp_path))
    assert m2.retrieve("Nicholas")
