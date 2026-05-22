# Field Lifecycle in Virgo-2

Virgo-2 lifecycle management keeps neural-memory fields durable, queryable, and incrementally improvable without introducing heavyweight orchestration.

## Core components

## `FieldVault`

`FieldVault` is the persistence layer for per-field artifacts. Each field stores:

- `field.npz` (neural field weights + numeric artifacts)
- `records.tsv` (human-readable memory records)

The vault can create missing field directories, load existing fields, and save updated fields.

## `FieldRegistry`

`FieldRegistry` tracks metadata about fields:

- field names and paths,
- record counts,
- dirty/refit state,
- timestamps and health-relevant status.

Registry state is the system-of-record for what exists and what needs maintenance.

## `FieldLifecycleManager`

`FieldLifecycleManager` orchestrates ingest/retrieve/update flow:

1. Route incoming text via `SemanticTaxonomy`.
2. Load/create target memory field from vault.
3. Append record and update salience.
4. Mark metadata in registry (count, dirty).
5. Fit/refit neural field as needed.
6. Save artifacts back to vault and registry.

## Default fields

By default, lifecycle initialization creates/uses these conceptual fields:

- `conversation_core`
- `identity_core`
- `project_core`
- `semantic_core`
- `procedural_core`

This keeps conversational memories separated by role while remaining in one vault.

## Dirty/refit behavior

When new records are ingested, fields become `dirty`. Dirty means the records changed and the fitted field should be refreshed. Lifecycle operations trigger `fit()`/`refit` and clear dirty state after successful save.

## Save/load cycle

- Load: resolve field from registry path, then load from vault artifacts.
- Mutate: add records, update salience, possibly reinforce retrieved records.
- Fit: recompute field approximation from latest records.
- Save: write `field.npz` + `records.tsv`, then persist registry metadata.

## Retrieval and reinforcement

`MultiFieldRetriever` runs query across multiple fields and returns ranked `RetrievedMemory` results. Retrieved memories receive salience reinforcement so frequently-useful memories stay retrievable.

This closes the loop between use and retention: retrieval influences future prominence.
