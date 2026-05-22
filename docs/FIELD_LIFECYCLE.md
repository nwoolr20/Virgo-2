# Field Lifecycle
Fields are compact neural memory surfaces stored in a vault with a TSV registry.

- `FieldVault` stores per-field `field.npz` + `records.tsv`.
- `FieldRegistry` tracks kind/path/count/dirty/lineage.
- `FieldLifecycleManager` ingests text, routes with taxonomy, fits/saves fields, and retrieves/reinforces.
- Fields are auto-created on first use.
- Dirty flags indicate refit/save needs.
