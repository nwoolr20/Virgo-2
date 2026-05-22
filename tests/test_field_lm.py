from pathlib import Path

import numpy as np

from virgo2.lm.field_lm import NeuralFieldLanguageModel


def test_overfit_and_save_load(tmp_path) -> None:
    model = NeuralFieldLanguageModel(seed=0)
    model.fit_texts(["hellohello"], epochs=1)
    logits = model.predict_next("hell")
    assert logits.shape[1] == model.codec.vocab_size
    model.save(tmp_path)
    loaded = NeuralFieldLanguageModel.load(tmp_path)
    assert loaded.predict_next("hell").shape == logits.shape


def test_predict_next_coordinates_change_logits() -> None:
    model = NeuralFieldLanguageModel(seed=0)
    model.fit_texts(["hello world hello moon"], epochs=1)
    short = model.predict_next("hello")
    long = model.predict_next("hello world")
    different_context_same_len_a = model.predict_next("abcde")
    different_context_same_len_b = model.predict_next("vwxyz")
    assert not np.allclose(short, long)
    assert not np.allclose(different_context_same_len_a, different_context_same_len_b)


def test_checkpoint_validation(tmp_path: Path) -> None:
    model = NeuralFieldLanguageModel(seed=0)
    model.fit_texts(["hello world"], epochs=1)
    model.save(tmp_path)
    ok, msg = NeuralFieldLanguageModel.validate_checkpoint(tmp_path)
    assert ok, msg
