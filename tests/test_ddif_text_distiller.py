from virgo2.ddif.distiller import TextFieldDistiller


def test_distiller_roundtrip(tmp_path) -> None:
    d = TextFieldDistiller(seed=0)
    d.fit_text("hello world", epochs=200)
    d.save(tmp_path)
    text = d.sample("he", max_chars=10)
    assert text
