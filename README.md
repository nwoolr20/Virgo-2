# Virgo-2

Virgo-2 is a focused neural-field memory and language modeling research project. It treats memory as a continuous function over coordinates instead of only as discrete token embeddings, and layers lifecycle management plus lightweight conversational memory on top.

## What Virgo-2 is

- A **core neural-field memory substrate** (`NeuralMemory`) with salience-weighted records and continuous retrieval.
- A **field lifecycle + conversational memory layer** (`FieldLifecycleManager`, `ConversationMemory`, `FieldVault`, `FieldRegistry`) for routing, persistence, reinforcement, and folding.
- An **experimental DDiF-inspired neural-field LM** for tiny text distillation and generation experiments.

## What Virgo-2 is not

- Not GPT-class model infrastructure.
- Not production AGI, autonomous agent platform, or cloud-scale serving stack.
- Not dependent on heavy mandatory components like FAISS, sentence-transformers, or hosted APIs.
- Not a GUI or Kubernetes/daemon-based platform.

## Why neural fields matter

Neural fields provide a compact, continuous representation where nearby coordinates can produce semantically related values. In Virgo-2 this supports:

- smooth interpolation over memory space,
- salience-aware retrieval and reinforcement,
- portable storage (`field.npz` + `records.tsv`) that stays lightweight.

## Three-layer architecture

1. **Core substrate**: `CoordinateEncoder`, `NeuralField`, `NeuralMemory`.
2. **Lifecycle + conversation**: taxonomy routing, vault persistence, registry metadata, multi-field retrieval, salience reinforcement, and consolidation/folding.
3. **DDiF-inspired LM**: character-level coordinate mapping and tiny generation loop for experimental language modeling.

For deeper details see `docs/ARCHITECTURE.md` and layer-specific docs in `docs/`.

## CLI quickstart

```bash
python -m pip install -e ".[dev]"

virgo2 ingest sample.txt ./tmp_memory
virgo2 query ./tmp_memory "continuous memory field" --k 5
virgo2 export-browser ./tmp_memory ./tmp_browser_bundle

virgo2 vault-init ./tmp_vault
virgo2 remember ./tmp_vault "Remember that Virgo-2 is a neural-field language model project."
virgo2 recall ./tmp_vault "What is Virgo-2?"
virgo2 chat-memory ./tmp_vault "Do you remember what I am building?"

virgo2 fold ./tmp_vault conversation_core folded_conversation_0001 --max-records 5
virgo2 forge-check ./tmp_vault --report ./tmp_vault/report.md

virgo2 lm-train tiny_lm.txt ./tmp_char_lm --epochs 20
virgo2 lm-generate ./tmp_char_lm "hello" --max-chars 40
```

## Python usage examples

### Core memory substrate

```python
from virgo2 import NeuralMemory

memory = NeuralMemory()
memory.add("Neural fields store memory as continuous functions.")
memory.add("Virgo-2 is a research-grade system.")
memory.fit()

results = memory.retrieve("continuous memory", k=3)
for record, score in results:
    print(score, record.text)
```

### Vault + lifecycle example

```python
from virgo2 import FieldLifecycleManager, FieldRegistry, FieldVault

vault = FieldVault("./tmp_vault")
registry = FieldRegistry("./tmp_vault")
manager = FieldLifecycleManager(vault=vault, registry=registry)

manager.initialize_defaults()
manager.ingest("Remember that my name is Nicholas.")
manager.ingest("Virgo-2 is a neural-field LM experiment.")

hits = manager.retrieve("What is Virgo-2?", k=5)
for hit in hits:
    print(hit.field_name, hit.score, hit.record.text)
```

### Field folding example

```python
from virgo2.consolidation import FieldConsolidator

consolidator = FieldConsolidator(vault, registry)
consolidator.fold_field("conversation_core", "folded_conversation_0001", max_records=250)
registry.save()
```

### ForgeLite validation example

```python
from virgo2.forge import ForgeLite

forge = ForgeLite(vault, registry)
report = forge.run_checks()
print(report)
forge.write_report("./tmp_vault/report.md")
```

### Browser export example

```python
from virgo2.browser_export import export_browser_bundle
from virgo2.storage import load_memory

memory = load_memory("./tmp_memory")
export_browser_bundle(memory, "./tmp_browser_bundle")
```

## Limitations

- Virgo-2 is not GPT-class in reasoning or generation quality.
- The DDiF-inspired LM is early-stage and character-level.
- Field folding is deterministic/simple summarization, not learned compression.
- Memory lifecycle is functional but still research-grade.

## Roadmap

See `docs/ROADMAP.md` for concrete milestones around:

- stronger retrieval scoring and field health metrics,
- smarter consolidation and optional neural compression,
- improved LM training/evaluation and prompt-conditioning.
