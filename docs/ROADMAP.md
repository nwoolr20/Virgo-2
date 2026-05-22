# Virgo-2 Roadmap

## Near-term (stabilization)

- Expand lifecycle and conversation regression tests for edge cases (dirty-state, missing artifacts, fold/merge interactions).
- Add richer ForgeLite checks and actionable repair hints.
- Improve retrieval scoring diagnostics and traceability.

## Mid-term (memory quality)

- Add configurable consolidation policies (thresholds, scheduled folds, retention windows).
- Improve salience dynamics (decay + reinforcement balancing).
- Introduce optional learned compression prototype for folded fields.

## Mid-term (LM experimentation)

- Improve DDiF-inspired text distillation losses and sampling controls.
- Add reproducible training/eval scripts for tiny corpora.
- Expand context-window and conditioning experiments.

## Long-term (research direction)

- Evaluate unified memory+LM coupling where field memories condition generation.
- Compare deterministic fold vs learned compression under retrieval-quality budgets.
- Document empirical findings with reproducible experiment cards.


Updated for stable-release hardening pass (0.9.0 target).


## Neural-field LM status (required core)
Virgo-2 targets a small but powerful neural-field language model. DDiF dataset-to-field distillation and the LM stack are **required core infrastructure** (beta maturity), not optional sidecars. Stable scope remains memory substrate, vault/registry/lifecycle, and deterministic reflection/folding. Experimental scope includes only GPT/LLaMA-level coherence claims, multimodal expansion, and large-scale training claims.
