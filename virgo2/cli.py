from __future__ import annotations

import argparse
from pathlib import Path

from .browser_export import export_browser_bundle
from .consolidation import FieldConsolidator
from .conversation import ConversationMemory
from .curriculum import CurriculumQueue
from .ddif import TextFieldDistiller
from .field_builder import FieldBuildRequest, FieldBuilder
from .forge import ForgeLite
from .lifecycle import FieldLifecycleManager
from .memory import NeuralMemory
from .merge import merge_memories
from .reflection import ReflectionEngine
from .registry import FieldRegistry
from .session import SessionOverlay
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

    # Legacy commands
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

    # New automation commands
    create_field = sub.add_parser("create-field")
    create_field.add_argument("vault_dir")
    create_field.add_argument("field_name")
    create_field.add_argument("input_txt")
    create_field.add_argument("--type", default="text")

    auto_remember = sub.add_parser("auto-remember")
    auto_remember.add_argument("vault_dir")
    auto_remember.add_argument("text")

    process_message = sub.add_parser("process-message")
    process_message.add_argument("vault_dir")
    process_message.add_argument("message")
    process_message.add_argument("--session-id", default=None)

    maintenance_cycle = sub.add_parser("maintenance-cycle")
    maintenance_cycle.add_argument("vault_dir")
    maintenance_cycle.add_argument("--max-records", type=int, default=500)
    maintenance_cycle.add_argument("--no-auto-fold", action="store_true")

    session_start = sub.add_parser("session-start")
    session_start.add_argument("vault_dir")
    session_start.add_argument("--session-id", default=None)

    session_add = sub.add_parser("session-add")
    session_add.add_argument("vault_dir")
    session_add.add_argument("session_id")
    session_add.add_argument("role")
    session_add.add_argument("text")

    session_context = sub.add_parser("session-context")
    session_context.add_argument("vault_dir")
    session_context.add_argument("session_id")
    session_context.add_argument("query")

    session_fold = sub.add_parser("session-fold")
    session_fold.add_argument("vault_dir")
    session_fold.add_argument("session_id")

    reflect = sub.add_parser("reflect")
    reflect.add_argument("vault_dir")
    reflect.add_argument("field_name")
    reflect.add_argument("--auto-promote", action="store_true")
    reflect.add_argument("--auto-fold", action="store_true")

    curriculum_add = sub.add_parser("curriculum-add")
    curriculum_add.add_argument("vault_dir")
    curriculum_add.add_argument("text")
    curriculum_add.add_argument("--domain", default="general")
    curriculum_add.add_argument("--difficulty", type=int, default=1)

    curriculum_next = sub.add_parser("curriculum-next")
    curriculum_next.add_argument("vault_dir")
    curriculum_next.add_argument("--batch-size", type=int, default=10)

    status_cmd = sub.add_parser("status")
    status_cmd.add_argument("vault_dir")

    release_check = sub.add_parser("release-check")
    release_check.add_argument("vault_dir")
    release_check.add_argument("--report", default=None)

    registry_validate = sub.add_parser("registry-validate")
    registry_validate.add_argument("vault_dir")

    taxonomy_classify = sub.add_parser("taxonomy-classify")
    taxonomy_classify.add_argument("text")

    args = parser.parse_args()

    if args.cmd == "vault-init":
        manager = _manager(args.vault_dir)
        manager.initialize_defaults()
        manager.save_all()
        print(f"Vault initialized at {args.vault_dir}")
    elif args.cmd == "remember":
        manager = _manager(args.vault_dir)
        rec = manager.ingest(args.text, field_name=args.field)
        print(f"stored field={args.field or manager.taxonomy.field_for(args.text)} salience={rec.salience:.3f}")
    elif args.cmd == "recall":
        manager = _manager(args.vault_dir)
        for r in manager.retrieve(args.query, k=args.k):
            print(f"{r.rank}. [{r.field_name}] score={r.score:.4f} text={r.record.text}")
    elif args.cmd == "chat-memory":
        manager = _manager(args.vault_dir)
        conv = ConversationMemory(manager)
        conv.converse_memory_update(args.user_message)
        print(conv.context_for(args.user_message))
    elif args.cmd == "fold":
        manager = _manager(args.vault_dir)
        out = FieldConsolidator(manager.vault, manager.registry).fold_field(args.source_field, args.target_field, max_records=args.max_records)
        print(f"folded to {args.target_field} records={len(out.records)}")
    elif args.cmd == "merge-fields":
        manager = _manager(args.vault_dir)
        out = FieldConsolidator(manager.vault, manager.registry).merge_fields(args.source_fields, args.target_field)
        print(f"merged to {args.target_field} records={len(out.records)}")
    elif args.cmd == "forge-check":
        manager = _manager(args.vault_dir)
        forge = ForgeLite(manager.vault, manager.registry)
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
        memories = [load_memory(path) for path in args.stores]
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
    elif args.cmd == "create-field":
        manager = _manager(args.vault_dir)
        lines = [line for line in Path(args.input_txt).read_text(encoding="utf-8").splitlines() if line.strip()]
        request = FieldBuildRequest(name=args.field_name, text_items=lines, field_type=args.type)
        result = FieldBuilder(manager.vault, manager.registry, manager.taxonomy).create_field(request)
        print(result)
    elif args.cmd == "auto-remember":
        manager = _manager(args.vault_dir)
        print(manager.ingest_auto(args.text))
    elif args.cmd == "process-message":
        manager = _manager(args.vault_dir)
        print(ConversationMemory(manager).process_user_message(args.message, session_id=args.session_id))
    elif args.cmd == "maintenance-cycle":
        manager = _manager(args.vault_dir)
        print(manager.maintenance_cycle(max_records_before_fold=args.max_records, auto_fold=not args.no_auto_fold))
    elif args.cmd == "session-start":
        session = SessionOverlay(_manager(args.vault_dir), session_id=args.session_id)
        print({"session_id": session.info.session_id, "field_name": session.info.field_name})
    elif args.cmd == "session-add":
        session = SessionOverlay(_manager(args.vault_dir), session_id=args.session_id)
        print(session.automate_after_turn(args.role, args.text))
    elif args.cmd == "session-context":
        session = SessionOverlay(_manager(args.vault_dir), session_id=args.session_id)
        print(session.retrieve_context(args.query))
    elif args.cmd == "session-fold":
        session = SessionOverlay(_manager(args.vault_dir), session_id=args.session_id)
        print(session.fold_session())
    elif args.cmd == "reflect":
        manager = _manager(args.vault_dir)
        report = ReflectionEngine(manager).reflect_on_field(args.field_name, auto_promote=args.auto_promote, auto_fold=args.auto_fold)
        print(report)
    elif args.cmd == "curriculum-add":
        queue = CurriculumQueue(Path(args.vault_dir) / "curriculum.tsv")
        print(queue.add(args.text, domain=args.domain, difficulty=args.difficulty))
    elif args.cmd == "curriculum-next":
        queue = CurriculumQueue(Path(args.vault_dir) / "curriculum.tsv")
        queue.load()
        print(queue.next_batch(batch_size=args.batch_size))
    elif args.cmd == "status":
        manager = _manager(args.vault_dir)
        print(manager.status())
    elif args.cmd == "release-check":
        manager = _manager(args.vault_dir)
        forge = ForgeLite(manager.vault, manager.registry)
        result = forge.release_check()
        print(result)
        if args.report:
            Path(args.report).write_text("# Virgo-2 Release Check\n\n" + str(result), encoding="utf-8")
            print(f"report={args.report}")
    elif args.cmd == "registry-validate":
        manager = _manager(args.vault_dir)
        print(manager.registry.validate())
    elif args.cmd == "taxonomy-classify":
        from dataclasses import asdict
        from .taxonomy import SemanticTaxonomy

        print(asdict(SemanticTaxonomy().classify(args.text)))


if __name__ == "__main__":
    main()
