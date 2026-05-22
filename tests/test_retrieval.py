from virgo2.memory import NeuralMemory
from virgo2.retrieval import MultiFieldRetriever


def test_multi_field_retrieval() -> None:
    a = NeuralMemory()
    a.add("Virgo project memory", salience=2.0)
    a.fit()

    b = NeuralMemory()
    b.add("Virgo project memory", salience=1.0)
    b.add("other note")
    b.fit()

    out = MultiFieldRetriever({"a": a, "b": b}).search("Virgo", k=5)
    assert out
    assert out[0].rank == 1
    texts = [x.record.text for x in out]
    assert texts.count("Virgo project memory") == 1
