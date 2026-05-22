from virgo2.memory import NeuralMemory
from virgo2.storage import load_memory, save_memory


def test_save_load_roundtrip(tmp_path) -> None:
    m = NeuralMemory()
    m.add("hello", metadata={"kind": "greeting"})
    m.fit()
    save_memory(m, tmp_path / "store")
    loaded = load_memory(tmp_path / "store")
    assert loaded.records[0].text == "hello"
    assert loaded.retrieve("hello", k=1)


def test_storage_creates_dir(tmp_path) -> None:
    m = NeuralMemory()
    m.add("x")
    m.fit()
    save_memory(m, tmp_path / "nested" / "store")
    assert (tmp_path / "nested" / "store" / "field.npz").exists()
