from virgo2.registry import FieldRegistry


def test_registry_roundtrip(tmp_path) -> None:
    r = FieldRegistry(tmp_path)
    r.register("a", "semantic", tmp_path / "fields/a", description="line\nwith\ttab")
    r.mark_dirty("a", True)
    r.update_count("a", 3)
    r.set_folded_from("a", ["x", "y"])
    r.save()

    r2 = FieldRegistry(tmp_path)
    r2.load()
    a = r2.get("a")
    assert a and a.dirty and a.record_count == 3 and a.folded_from == ["x", "y"]
    assert a.description == "line\nwith\ttab"


def test_registry_legacy_load(tmp_path) -> None:
    p = tmp_path / "registry.tsv"
    p.write_text(
        "name\tkind\tpath\trecord_count\tcreated_at\tupdated_at\tdirty\tfolded_from\n"
        "old\tsemantic\t/tmp/old\t2\t\t\t0\t\n",
        encoding="utf-8",
    )
    r = FieldRegistry(tmp_path)
    r.load()
    assert r.get("old") is not None
