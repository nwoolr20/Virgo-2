# Virgo-2 Architecture

Virgo-2 is a three-layer research architecture centered on neural-field memory.

## Layer 1: Core neural-field memory substrate

Primary modules:

- `virgo2/coordinates.py` (`CoordinateEncoder`)
- `virgo2/field.py` (`NeuralField`)
- `virgo2/memory.py` (`MemoryRecord`, `NeuralMemory`)
- `virgo2/storage.py` (save/load to `field.npz` + `records.tsv`)

Responsibilities:

- Convert text to stable coordinates.
- Fit a continuous field representation over memory records.
- Retrieve records by coordinate similarity + salience.
- Persist memory to local artifacts.

## Layer 2: Lifecycle + conversational memory

Primary modules:

- `registry.py`, `vault.py`, `lifecycle.py`
- `taxonomy.py`, `retrieval.py`, `salience.py`
- `conversation.py`, `consolidation.py`, `forge.py`

Responsibilities:

- Route text into semantic fields.
- Load/create/save fields with metadata tracking.
- Retrieve across fields and reinforce useful memories.
- Build compact context packs for conversational use.
- Merge/fold bloated fields.
- Validate system health using ForgeLite checks.

## Layer 3: Experimental DDiF-inspired LM

Primary modules:

- `virgo2/ddif/`
- `virgo2/lm/`
- `virgo2/training/`

Responsibilities:

- Distill text statistics into compact field-like representations.
- Train a tiny character-level neural-field LM.
- Generate short text from prompts for research experiments.

## End-to-end flow

User text -> `ConversationMemory.add_turn()` -> `FieldLifecycleManager.ingest()` -> `SemanticTaxonomy` routing -> `FieldVault` load/create -> `NeuralMemory` update -> `FieldRegistry` dirty/count update -> fit/refit -> vault save -> cross-field retrieval/reinforcement -> `ConversationMemory.context_for()` -> optional fold/merge -> `ForgeLite` validation.

## Design constraints

- Keep implementation local-first and lightweight.
- Avoid platform sprawl (no mandatory cloud APIs, orchestration systems, or heavyweight dependencies).
- Maintain honest research framing: functional but non-production and non-GPT-class.
