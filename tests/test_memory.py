from virgo2.memory import NeuralMemory
from virgo2.merge import merge_memories


def test_add_fit_retrieve() -> None:
    m = NeuralMemory()
    m.add("neural fields store memories")
    m.add("cats are animals")
    m.fit()
    res = m.retrieve("neural fields memory", k=1)
    assert res and "neural" in res[0][0].text


def test_empty_retrieve() -> None:
    assert NeuralMemory().retrieve("anything") == []


def test_salience_and_decay() -> None:
    m = NeuralMemory()
    m.add("topic one", salience=0.2)
    m.add("topic one", salience=3.0)
    m.fit()
    top = m.retrieve("topic one", k=1)[0][0]
    assert top.salience == 3.0
    m.decay(0.1)
    assert m.records[1].salience < 3.0


def test_merge_combines_records() -> None:
    a = NeuralMemory()
    a.add("a")
    a.fit()
    b = NeuralMemory()
    b.add("b")
    b.fit()
    c = merge_memories([a, b])
    assert len(c.records) == 2
