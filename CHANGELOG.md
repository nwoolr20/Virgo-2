## 0.9.0
### Added
- Release-check surfaces for lifecycle and ForgeLite.
- Registry validation API and CLI access.

### Changed
- Deterministic weighted taxonomy classification with confidence and multi-tags.
- Reflection report expanded with deterministic counters and serialization.

### Fixed
- Reflection duplicate promotion guard.
- Registry malformed row tracking.

### Verified
- Lint, pytest, and CLI smoke workflows.

### Known Limitations
- Experimental neural-field LM remains non-production.
- Deterministic consolidation is not LLM semantic summarization.


## Neural-field LM status (required core)
Virgo-2 targets a small but powerful neural-field language model. DDiF dataset-to-field distillation and the LM stack are **required core infrastructure** (beta maturity), not optional sidecars. Stable scope remains memory substrate, vault/registry/lifecycle, and deterministic reflection/folding. Experimental scope includes only GPT/LLaMA-level coherence claims, multimodal expansion, and large-scale training claims.
