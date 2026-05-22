from virgo2.cli import main
from virgo2.ddif.distiller import TextFieldDistiller


def test_chat_smoke_context_and_writeback(tmp_path, monkeypatch, capsys) -> None:
    corpus = "hello memory\nVirgo stores memory"
    model_dir = tmp_path / "model"
    vault_dir = tmp_path / "vault"

    distiller = TextFieldDistiller(seed=0)
    distiller.fit_text(corpus, epochs=1)
    distiller.save(model_dir)

    monkeypatch.setattr("sys.argv", ["virgo2", "vault-init", str(vault_dir)])
    main()
    monkeypatch.setattr("sys.argv", ["virgo2", "remember", str(vault_dir), "Virgo stores memory"])
    main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "virgo2",
            "chat",
            str(vault_dir),
            str(model_dir),
            "hello memory",
            "--session-id",
            "s1",
            "--max-chars",
            "40",
            "--seed",
            "1",
        ],
    )
    main()
    out = capsys.readouterr().out
    assert "retrieved_context_summary" in out
    assert "generation_metrics" in out
    assert "storage_result" in out

    assert "'writeback_success': True" in out
