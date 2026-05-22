from pathlib import Path

from virgo2.lm.field_lm import NeuralFieldLanguageModel
from virgo2.training.evaluate import evaluate_model


def test_deterministic_and_seeded_generation() -> None:
    model = NeuralFieldLanguageModel(seed=0)
    model.fit_texts(["hello world"], epochs=1)
    a = model.generate("he", max_chars=20, deterministic=True)
    b = model.generate("he", max_chars=20, deterministic=True)
    assert a == b
    c = model.generate("he", max_chars=20, seed=7)
    d = model.generate("he", max_chars=20, seed=7)
    assert c == d


def test_evaluation_metrics_present() -> None:
    model = NeuralFieldLanguageModel(seed=0)
    model.fit_texts(["hello world\nhello moon"], epochs=1)
    metrics = evaluate_model(model, "hello world")
    for key in [
        "loss",
        "next_char_accuracy",
        "char_reconstruction_accuracy",
        "repetition_rate",
        "invalid_character_rate",
        "deterministic_repeatability",
        "sample_output",
        "evaluated_character_count",
        "weight_count",
        "feature_count",
        "codec_vocab_size",
        "checkpoint_roundtrip_success",
        "checkpoint_validation_message",
    ]:
        assert key in metrics


def test_invalid_checkpoint_handling(tmp_path: Path) -> None:
    model_dir = tmp_path / "bad_model"
    model_dir.mkdir()
    ok, _ = NeuralFieldLanguageModel.validate_checkpoint(model_dir)
    assert not ok
