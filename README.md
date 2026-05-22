# Virgo-2

Virgo-2 is a lightweight, CPU-first neural-field system with two layers:
- a **stable memory substrate** for retrieval and storage
- an **experimental DDiF-inspired neural-field language model (LM)**

## What it is
- Deterministic text -> coordinates encoder.
- Continuous neural field over coordinates.
- Salience-aware memory retrieval.
- Experimental coordinate-based character LM for reconstruction/generation.

## What it is not
- Not competitive with transformer LLMs.
- Not a transformer replacement today.
- Not dependent on Torch/Hugging Face for base usage.

## DDiF-inspired language adaptation
DDiF stores dataset information in neural fields via coordinate-to-quantity mappings. Virgo-2 adapts this idea to language:
- coordinates represent sequence position/context
- quantities represent character IDs/distributions (and later latent vectors)

This repository now separates:
1. `virgo2` core memory substrate
2. `virgo2.ddif` language distillation layer
3. `virgo2.lm` neural-field language model layer
4. `virgo2.training` CPU-safe training utilities
5. `virgo2.browser_export` browser runtime preparation

## Install
```bash
python -m pip install -e ".[dev]"
# optional training extras
python -m pip install -e ".[dev,train]"
```

## CLI quickstart
```bash
virgo2 ingest sample.txt ./tmp_memory
virgo2 query ./tmp_memory "continuous memory field" --k 5
virgo2 lm-train examples/tiny_corpus.txt ./tmp_char_lm --epochs 50
virgo2 lm-generate ./tmp_char_lm "hello" --max-chars 50
virgo2 ddif-reconstruct examples/tiny_corpus.txt ./tmp_ddif
virgo2 ddif-sample ./tmp_ddif --prompt "hello"
```

See `docs/DDIF_LANGUAGE_ADAPTATION.md` and `docs/NEURAL_FIELD_LM.md` for details.
