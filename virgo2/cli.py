from __future__ import annotations

import argparse
from pathlib import Path

from .browser_export import export_browser_bundle
from .consolidation import FieldConsolidator
from .conversation import ConversationMemory
from .ddif import TextFieldDistiller
from .forge import ForgeLite
from .lifecycle import FieldLifecycleManager
from .memory import NeuralMemory
from .merge import merge_memories
from .registry import FieldRegistry
from .storage import load_memory, save_memory
from .vault import FieldVault


def _load_or_new(path: str) -> NeuralMemory:
    p = Path(path)
    if (p / "field.npz").exists() and (p / "records.tsv").exists():
        return load_memory(p)
    return NeuralMemory()


def _manager(vault_dir: str) -> FieldLifecycleManager:
    return FieldLifecycleManager(vault=FieldVault(vault_dir), registry=FieldRegistry(vault_dir))


def main() -> None:
    parser = argparse.ArgumentParser(prog="virgo2")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for cmd in ["ingest", "query", "add", "decay", "merge", "export-browser", "inspect"]:
        sub.add_parser(cmd)
    ingest = sub.choices["ingest"]
    ingest.add_argument("input_txt")
    ingest.add_argument("store_dir")
    query = sub.choices["query"]
    query.add_argument("store_dir")
    query.add_argument("query")
    query.add_argument("--k", type=int, default=5)
    add = sub.choices["add"]
    add.add_argument("store_dir")
    add.add_argument("text")
    decay = sub.choices["decay"]
    decay.add_argument("store_dir")
    decay.add_argument("--rate", type=float, default=0.01)
    merge = sub.choices["merge"]
    merge.add_argument("output_store")
    merge.add_argument("stores", nargs="+")
    export = sub.choices["export-browser"]
    export.add_argument("store_dir")
    export.add_argument("output_dir")
    sub.choices["inspect"].add_argument("store_dir")

    lm_train = sub.add_parser("lm-train")
    lm_train.add_argument("input_txt")
    lm_train.add_argument("model_dir")
    lm_train.add_argument("--epochs", type=int, default=200)
    lm_generate = sub.add_parser("lm-generate")
    lm_generate.add_argument("model_dir")
    lm_generate.add_argument("prompt")
    lm_generate.add_argument("--max-chars", type=int, default=200)
    ddif_reconstruct = sub.add_parser("ddif-reconstruct")
    ddif_reconstruct.add_argument("input_txt")
    ddif_reconstruct.add_argument("output_dir")
    ddif_sample = sub.add_parser("ddif-sample")
    ddif_sample.add_argument("output_dir")
    ddif_sample.add_argument("--prompt", default="hello")
    ddif_sample.add_argument("--max-chars", type=int, default=120)

    vault_init = sub.add_parser("vault-init")
    vault_init.add_argument("vault_dir")
    remember = sub.add_parser("remember")
    remember.add_argument("vault_dir")
    remember.add_argument("text")
    remember.add_argument("--field", default=None)
    recall = sub.add_parser("recall")
    recall.add_argument("vault_dir")
    recall.add_argument("query")
    recall.add_argument("--k", type=int, default=8)
    chat_memory = sub.add_parser("chat-memory")
    chat_memory.add_argument("vault_dir")
    chat_memory.add_argument("user_message")
    fold = sub.add_parser("fold")
    fold.add_argument("vault_dir")
    fold.add_argument("source_field")
    fold.add_argument("target_field")
    fold.add_argument("--max-records", type=int, default=250)
    merge_fields = sub.add_parser("merge-fields")
    merge_fields.add_argument("vault_dir")
    merge_fields.add_argument("target_field")
    merge_fields.add_argument("source_fields", nargs="+")
    forge_check = sub.add_parser("forge-check")
    forge_check.add_argument("vault_dir")
    forge_check.add_argument("--report", default=None)

    args = parser.parse_args()
    if args.cmd == "vault-init":
        m = _manager(args.vault_dir)
        m.initialize_defaults()
        m.save_all()
        print(f"Vault initialized at {args.vault_dir}")
    elif args.cmd == "remember":
        m = _manager(args.vault_dir)
        rec = m.ingest(args.text, field_name=args.field)
        print(f"stored field={args.field or m.taxonomy.field_for(args.text)} salience={rec.salience:.3f}")
    elif args.cmd == "recall":
        m = _manager(args.vault_dir)
        for r in m.retrieve(args.query, k=args.k):
            print(f"{r.rank}. [{r.field_name}] score={r.score:.4f} text={r.record.text}")
    elif args.cmd == "chat-memory":
        m = _manager(args.vault_dir)
        c = ConversationMemory(m)
        c.converse_memory_update(args.user_message)
        print(c.context_for(args.user_message))
    elif args.cmd == "fold":
        m = _manager(args.vault_dir)
        out = FieldConsolidator(m.vault, m.registry).fold_field(
            args.source_field,
            args.target_field,
            max_records=args.max_records,
        )
        m.registry.save()
        print(f"folded to {args.target_field} records={len(out.records)}")
    elif args.cmd == "merge-fields":
        m = _manager(args.vault_dir)
        out = FieldConsolidator(m.vault, m.registry).merge_fields(args.source_fields, args.target_field)
        m.registry.save()
        print(f"merged to {args.target_field} records={len(out.records)}")
    elif args.cmd == "forge-check":
        m = _manager(args.vault_dir)
        forge = ForgeLite(m.vault, m.registry)
        print(forge.run_checks())
        if args.report:
            forge.write_report(args.report)
            print(f"report={args.report}")
    elif args.cmd == "ingest":
        memory = NeuralMemory()
        for line in Path(args.input_txt).read_text(encoding="utf-8").splitlines():
            if line.strip():
                memory.add(line.strip())
        memory.fit()
        save_memory(memory, args.store_dir)
        print(f"Ingested {len(memory.records)} records")
    elif args.cmd == "query":
        memory = load_memory(args.store_dir)
        for i, (rec, score) in enumerate(memory.retrieve(args.query, k=args.k), 1):
            print(f"{i}. score={score:.4f} salience={rec.salience:.3f} text={rec.text}")
    elif args.cmd == "add":
        memory = _load_or_new(args.store_dir)
        memory.add(args.text)
        memory.fit()
        save_memory(memory, args.store_dir)
        print("Record added")
    elif args.cmd == "decay":
        memory = load_memory(args.store_dir)
        memory.decay(args.rate)
        memory.fit()
        save_memory(memory, args.store_dir)
        print("Decay applied")
    elif args.cmd == "merge":
        memories = [load_memory(p) for p in args.stores]
        merged = merge_memories(memories)
        save_memory(merged, args.output_store)
        print(f"Merged {len(memories)} stores into {args.output_store}")
    elif args.cmd == "export-browser":
        memory = load_memory(args.store_dir)
        export_browser_bundle(memory, args.output_dir)
        print(f"Exported browser bundle to {args.output_dir}")
    elif args.cmd == "inspect":
        memory = load_memory(args.store_dir)
        print(f"records={len(memory.records)}")
        print(f"field_fitted={memory.field.weights is not None}")
        print(f"dimensions={memory.encoder.dimensions}")
    elif args.cmd == "lm-train":
        text = Path(args.input_txt).read_text(encoding="utf-8")
        distiller = TextFieldDistiller(seed=0)
        distiller.fit_text(text, epochs=args.epochs)
        distiller.save(args.model_dir)
        print(f"Saved neural-field LM to {args.model_dir}")
    elif args.cmd == "lm-generate":
        model = TextFieldDistiller()
        model.model = model.model.load(args.model_dir)
        print(model.sample(args.prompt, max_chars=args.max_chars))
    elif args.cmd == "ddif-reconstruct":
        text = Path(args.input_txt).read_text(encoding="utf-8")
        distiller = TextFieldDistiller(seed=0)
        distiller.fit_text(text, epochs=200)
        distiller.save(args.output_dir)
        print(f"Distilled text field to {args.output_dir}")
    elif args.cmd == "ddif-sample":
        model = TextFieldDistiller()
        model.model = model.model.load(args.output_dir)
        print(model.sample(args.prompt, max_chars=args.max_chars))


if __name__ == "__main__":
    main()
