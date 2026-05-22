from virgo2.lifecycle import FieldLifecycleManager
from virgo2.registry import FieldRegistry
from virgo2.vault import FieldVault


def test_reflection_report_and_determinism(tmp_path):
    manager = FieldLifecycleManager(FieldVault(tmp_path), FieldRegistry(tmp_path))
    manager.ingest("my name is Alex")
    manager.ingest("my name is Alex")
    report1 = manager.reflection_engine.reflect_on_field("identity_core", auto_promote=False, auto_fold=False)
    report2 = manager.reflection_engine.reflect_on_field("identity_core", auto_promote=False, auto_fold=False)
    assert report1.to_dict() == report2.to_dict()
    assert report1.inspected_record_count >= 2
    assert report1.promoted_count == 0
