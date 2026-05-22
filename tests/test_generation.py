from virgo2.lm.field_lm import NeuralFieldLanguageModel


def test_generation_non_empty() -> None:
    model = NeuralFieldLanguageModel(seed=0)
    model.fit_texts(["hello world"], epochs=200)
    text = model.generate("he", max_chars=10)
    assert len(text) > 2
