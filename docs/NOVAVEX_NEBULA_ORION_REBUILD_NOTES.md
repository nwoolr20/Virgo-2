# NOVAVEX / Nebula / Orion Rebuild Notes (Virgo-2 Scope)

This document clarifies what conceptual pieces were retained from broader historical ideas, and what was intentionally excluded to keep Virgo-2 focused.

## Concepts rebuilt from FLEX

- Lightweight composability of small modules.
- Local-first, scriptable workflows.
- Fast experimentation over heavy infrastructure.

## Concepts rebuilt from FORGE

- Structural integrity checks (`ForgeLite`) over storage + metadata.
- Inspectability of state (registry, field artifacts, health report generation).
- Deterministic validation pathways rather than opaque runtime magic.

## Concepts rebuilt from STL

- Emphasis on compact, testable primitives.
- Explicit data flow between components.
- Reproducible CLI and Python APIs for iterative research loops.

## Concepts rebuilt from Nebula

- Multi-field memory organization by semantic role.
- Retrieval-driven reinforcement and long-lived memory shaping.
- Field lifecycle thinking (ingest, update, consolidate, validate).

## Concepts rebuilt from Orion

- Experimental language modeling as a separate but connected track.
- Coordinate-centric modeling ideas adapted into tiny practical prototypes.
- Iterative scientific framing over productized "AI assistant" framing.

## Intentionally excluded (and why)

Excluded from Virgo-2 by design:

- orchestration sprawl (daemons, clusters, Kubernetes),
- cloud-first dependencies and mandatory external APIs,
- monolithic "AGI OS" ambitions,
- emotional/social cognition layers,
- agent-swarm infrastructure.

Reason: these increase operational complexity without serving Virgo-2's current research objective: a clean neural-field memory system plus an experimental DDiF-inspired LM.
