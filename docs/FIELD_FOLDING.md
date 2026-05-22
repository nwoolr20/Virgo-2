# Field Folding
Field bloat occurs as records accumulate. Folding preserves high-salience records and summarizes lower-salience clusters.

- Sort by salience.
- Keep high-salience top slice.
- Summarize low-salience tail deterministically.
- Save folded target and lineage in registry (`folded_from`).

Limits: no LLM summarizer; summaries are heuristic and compact only.
