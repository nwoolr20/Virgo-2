# Virgo-2 Architecture

Virgo-2 maps text into deterministic coordinates, stores records with salience, and fits a continuous neural field that supports resonance-aware retrieval.

## Coordinate Encoder
`CoordinateEncoder` tokenizes text, hashes tokens with `blake2b`, projects through sinusoidal features, and normalizes output vectors.

## Continuous Field
`NeuralField` builds Fourier-style features from coordinates and fits ridge regression weights for compact continuous approximation.

## Memory Records and Salience
`NeuralMemory` stores `MemoryRecord{text, metadata, salience}`. Salience biases retrieval and decays over time.

## Retrieval Scoring
Retrieval combines cosine distance to stored coordinates with field resonance mismatch and salience boost into a single lower-is-better score.

## Persistence Format
Stores are directory-based:
- `field.npz`: encoder config, coordinates, field basis, and learned weights.
- `records.tsv`: text, salience, and metadata json per record.

## Browser Export
Browser export writes `field.npz`, `records.tsv`, `manifest.txt`, and a mini README for future JS/WebGPU loaders.

## Merge Behavior
`merge_memories` concatenates records from compatible memories and refits one merged field.

## Future Decoder Layer
A tiny optional decoder may be added later as an experimental module over this substrate.
