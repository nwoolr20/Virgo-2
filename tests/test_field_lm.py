from virgo2.lm.field_lm import NeuralFieldLanguageModel


def test_overfit_and_save_load(tmp_path) -> None:
    model = NeuralFieldLanguageModel(seed=0)
    model.fit_texts(["hellohello"], epochs=300)
    logits = model.predict_next("hell")
    assert logits.shape[1] == model.codec.vocab_size
    model.save(tmp_path)
    loaded = NeuralFieldLanguageModel.load(tmp_path)
    assert loaded.predict_next("hell").shape == logits.shape
