# Conversational Memory in Virgo-2

`ConversationMemory` provides a practical conversational layer over lifecycle-managed fields. It is designed for compact memory updates and context reconstruction, not chat-model orchestration.

## Field roles

Virgo-2 routes conversational content into semantically distinct fields:

- `conversation_core`: raw dialogue turns and short-term interaction history.
- `identity_core`: personal identifiers and stable self facts.
- `project_core`: mission, goals, scope, and project descriptors.
- `semantic_core`: conceptual/domain information and broad facts.
- `procedural_core`: instructions, workflows, and actionable steps.

## Turn storage

Each user message can be wrapped as a `ConversationTurn` and passed through `ConversationMemory.add_turn()`. The turn text is ingested through lifecycle manager logic, so storage is persisted in the vault-backed field memory.

## Fact extraction and routing

`SemanticTaxonomy` applies lightweight routing heuristics to map text to the most appropriate field. This avoids a monolithic memory bucket and improves retrieval precision.

## Context packs

`ConversationMemory.context_for(query)` retrieves ranked cross-field memories and builds a compact context pack suitable for downstream prompting or summarization.

A context pack intentionally emphasizes:

- relevance score,
- field provenance,
- compact memory snippets.

## On-the-fly learning

As queries retrieve memories, salience reinforcement updates the records. In effect, conversational usage acts as online memory sharpening without external retraining infrastructure.
