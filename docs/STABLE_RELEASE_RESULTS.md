# Virgo-2 Stable Release Results

## Summary
PASSED WITH LIMITATIONS

## Version Decision
0.9.0 due to remaining known limitations before 1.0.0 (experimental neural-field LM and deterministic non-LLM summarization).

## Commands Run
- python -m pip install -e ".[dev]"
- ruff check .
- python -m pytest
- virgo2 --help
- virgo2 vault-init /tmp/virgo2-smoke
- virgo2 status /tmp/virgo2-smoke
- virgo2 taxonomy-classify "my name is alex"
- virgo2 registry-validate /tmp/virgo2-smoke
- virgo2 release-check /tmp/virgo2-smoke

## Test Results
See command outputs from lint and pytest.

## CLI Smoke Results
All release CLI commands executed successfully.

## Release Check Results
Release-check command produced readiness dict.

## Registry Validation Results
Registry validation command produced validation dict.

## Public API Verification
Verified public exports import through `from virgo2 import *` coverage test.

## Limitations Remaining
- Experimental neural-field LM
- Deterministic, non-LLM consolidation summaries

## Final Recommendation
Ready for 0.9.0 tag; promote to 1.0.0 after remaining limitations are reduced.


## Neural-field LM status (required core)
Virgo-2 targets a small but powerful neural-field language model. DDiF dataset-to-field distillation and the LM stack are **required core infrastructure** (beta maturity), not optional sidecars. Stable scope remains memory substrate, vault/registry/lifecycle, and deterministic reflection/folding. Experimental scope includes only GPT/LLaMA-level coherence claims, multimodal expansion, and large-scale training claims.
