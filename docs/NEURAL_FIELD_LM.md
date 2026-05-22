# Neural Field LM

## Architecture
- `CharCodec`: deterministic char vocabulary.
- `TextCoordinateSet`: converts text to normalized position coordinates and targets.
- `NeuralFieldLanguageModel`: fits coordinate->logit mapping with a compact field backend.
- `TextFieldDistiller`: DDiF-flavored wrapper for reconstruction and sampling.

## Training objective
Given coordinates and next-character targets, learn logits minimizing cross-entropy (or reconstruction proxy for fallback field backends).

## Generation loop
1. Start from prompt.
2. Compute next coordinate from prompt length.
3. Predict logits.
4. Sample char with temperature.
5. Append and repeat.

## Limitations
- Tiny architecture proof only.
- No long-context transformer attention.
- Character-level modeling is low-capacity and brittle.

## Future browser/WebGPU path
- Export compact field parameters.
- Run inference in browser runtime/WebGPU kernels.
- Add multi-coordinate context features and latent targets.
