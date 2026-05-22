import pytest

from virgo2.ddif.distiller import TextFieldDistiller


def test_distiller_roundtrip(tmp_path) -> None:
    d = TextFieldDistiller(seed=0)
    d.fit_text("hello world", epochs=1)
    d.save(tmp_path)
    text = d.sample("he", max_chars=10)
    assert text


def test_distiller_empty_input_validation() -> None:
    d = TextFieldDistiller(seed=0)
    with pytest.raises(ValueError):
        d.fit_text("   ", epochs=1)


def test_fit_corpus_and_evaluate() -> None:
    d = TextFieldDistiller(seed=0)
    d.fit_corpus(["hello", "world"], epochs=1)
    metrics = d.evaluate("hello world", sample_prompt="h")
    assert "sample_output" in metrics
    assert "feature_count" in metrics
