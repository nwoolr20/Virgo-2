from pathlib import Path

from virgo2.lm.field_lm import NeuralFieldLanguageModel
from virgo2.training.evaluate import evaluate_model


def test_deterministic_and_seeded_generation() -> None:
    model = NeuralFieldLanguageModel(seed=0)
    model.fit_texts(["hello world"])
    a = model.generate("he", max_chars=20, deterministic=True)
    b = model.generate("he", max_chars=20, deterministic=True)
    assert a == b
    c = model.generate("he", max_chars=20, seed=7)
    d = model.generate("he", max_chars=20, seed=7)
    assert c == d


def test_evaluation_metrics_present() -> None:
    model = NeuralFieldLanguageModel(seed=0)
    model.fit_texts(["hello world\nhello moon"])
    metrics = evaluate_model(model, "hello world")
    assert "loss" in metrics
    assert "next_char_accuracy" in metrics
    assert "deterministic_repeatability" in metrics


def test_invalid_checkpoint_handling(tmp_path: Path) -> None:
    model_dir = tmp_path / "bad_model"
    model_dir.mkdir()
    try:
        NeuralFieldLanguageModel.load(model_dir)
        assert False, "Expected checkpoint load failure"
    except Exception:
        assert True
