# Virgo-2

Virgo-2 is a lightweight, CPU-first, browser-forward neural-field memory engine.

## What it is
- Deterministic text -> coordinates encoder.
- Continuous neural field over coordinates.
- Salience-aware memory retrieval.
- Compact portable storage and browser bundle export.

## What it is not
- Not a full trained LLM.
- Not a transformer replacement today.
- Not dependent on Torch, Hugging Face, FAISS, or sentence-transformers for base usage.

## Why neural-field memory
Virgo-2 explores whether language memory can be represented as a continuous coordinate-addressable field rather than static embedding tables, JSON prompt stuffing, or external vector DB infrastructure.

## Install
```bash
python -m pip install -e ".[dev]"
```

## CLI quickstart
```bash
virgo2 ingest sample.txt ./tmp_memory
virgo2 query ./tmp_memory "continuous memory field" --k 5
virgo2 inspect ./tmp_memory
virgo2 export-browser ./tmp_memory ./tmp_browser_bundle
```

## Python usage
```python
from virgo2.memory import NeuralMemory

mem = NeuralMemory()
mem.add("Neural fields store memory as continuous functions.")
mem.add("Browser deployment should stay lightweight.")
mem.fit()
print(mem.retrieve("continuous memory field", k=2))
```

## Architecture overview
1. Text is encoded into deterministic normalized coordinates.
2. Records are stored with metadata and salience.
3. A Fourier-style NumPy neural field is fit with ridge regression.
4. Retrieval combines cosine distance, field resonance, and salience.
5. Stores can be merged and exported for future browser runtimes.

See `docs/ARCHITECTURE.md` for full format and data-flow details.

## Roadmap
See `docs/ROADMAP.md` for phased milestones from core memory engine to optional decoder and evaluation suite.
