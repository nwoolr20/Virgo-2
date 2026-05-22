from __future__ import annotations

import argparse
from pathlib import Path

from .browser_export import export_browser_bundle
from .ddif import TextFieldDistiller
from .memory import NeuralMemory
from .merge import merge_memories
from .storage import load_memory, save_memory


def _load_or_new(path: str) -> NeuralMemory:
    p = Path(path)
    if (p / "field.npz").exists() and (p / "records.tsv").exists():
        return load_memory(p)
    return NeuralMemory()


def main() -> None:
    parser = argparse.ArgumentParser(prog="virgo2")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("input_txt")
    ingest.add_argument("store_dir")

    query = sub.add_parser("query")
    query.add_argument("store_dir")
    query.add_argument("query")
    query.add_argument("--k", type=int, default=5)

    add = sub.add_parser("add")
    add.add_argument("store_dir")
    add.add_argument("text")

    decay = sub.add_parser("decay")
    decay.add_argument("store_dir")
    decay.add_argument("--rate", type=float, default=0.01)

    merge = sub.add_parser("merge")
    merge.add_argument("output_store")
    merge.add_argument("stores", nargs="+")

    export = sub.add_parser("export-browser")
    export.add_argument("store_dir")
    export.add_argument("output_dir")

    inspect = sub.add_parser("inspect")
    inspect.add_argument("store_dir")

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

    args = parser.parse_args()

    if args.cmd == "ingest":
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
