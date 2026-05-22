from virgo2.browser_export import export_browser_bundle
from virgo2.memory import NeuralMemory


def test_export_creates_files(tmp_path) -> None:
    m = NeuralMemory()
    m.add("browser bundle")
    m.fit()
    export_browser_bundle(m, tmp_path / "bundle")
    assert (tmp_path / "bundle" / "manifest.txt").exists()
    assert (tmp_path / "bundle" / "records.tsv").exists()
    assert (tmp_path / "bundle" / "field.npz").exists()
