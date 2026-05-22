from virgo2.registry import FieldRegistry


def test_registry_roundtrip(tmp_path) -> None:
    r = FieldRegistry(tmp_path)
    r.register("a", "semantic", tmp_path / "fields/a")
    assert r.get("a") is not None
    assert len(r.list()) == 1
    r.mark_dirty("a", True)
    r.update_count("a", 3)
    r.set_folded_from("a", ["x", "y"])
    r.save()

    r2 = FieldRegistry(tmp_path)
    r2.load()
    a = r2.get("a")
    assert a and a.dirty and a.record_count == 3 and a.folded_from == ["x", "y"]
