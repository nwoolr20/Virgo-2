# Field Folding and Consolidation

Field consolidation prevents unbounded growth and keeps retrieval efficient in long-running usage.

## Why field bloat happens

Continuous ingestion adds records over time. Conversation-heavy workflows especially can create near-duplicate records, stale turns, and noisy tails that degrade retrieval signal.

## Merge vs fold

- **Merge**: combines complete records from multiple source fields into a target field.
- **Fold**: compresses a single large field into deterministic summary records, reducing count while preserving key content.

Use merge for topology changes; use fold for compaction.

## Salience preservation

Folding prioritizes higher-salience records when selecting/assembling summary output, so historically important memories are less likely to be lost.

## Deterministic summarization

Current folding is deterministic/simple, designed for repeatable compaction. It does not rely on external LLM APIs and does not pretend to be semantic-perfect summarization.

## Lineage tracking

Folded outputs should retain lineage metadata (source field identity and operation intent) through registry naming and operator conventions (for example, `folded_conversation_0001`).

## Current limitations

- No learned compression objective.
- Potential information loss in aggressive folds.
- Deterministic heuristics may miss subtle latent structure.

## Future neural compression

Planned upgrades include learned field distillation/compression passes that preserve retrieval quality under stronger size budgets while keeping local/offline operation.
