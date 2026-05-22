# Codex Completion Notes

- The prior repo state was internally inconsistent across retrieval-memory and generative-NFLM narratives.
- Old docs mixed incompatible claims about 6D/8D systems and completion status.
- This rebuild resolves direction around a single lightweight neural-field memory substrate.
- Heavy core dependencies (Torch, Hugging Face, FAISS, sentence-transformers, TextBlob, sklearn) were removed from default install.
- Generative LM work is now future/experimental scope, not part of Virgo-2 base runtime.
