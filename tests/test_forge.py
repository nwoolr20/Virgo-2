from virgo2.forge import ForgeLite
from virgo2.lifecycle import FieldLifecycleManager
from virgo2.registry import FieldRegistry
from virgo2.vault import FieldVault


def test_forge(tmp_path):
    m = FieldLifecycleManager(FieldVault(tmp_path), FieldRegistry(tmp_path))
    m.ingest("remember this fact")
    m.registry.mark_dirty("conversation_core", True)
    f = ForgeLite(m.vault, m.registry)
    checks = f.run_checks()
    assert isinstance(checks, dict)
    assert "dirty_fields" in checks and checks["loadable_fields_count"] >= 1
    report = tmp_path / "report.md"
    f.write_report(report)
    assert report.exists()
