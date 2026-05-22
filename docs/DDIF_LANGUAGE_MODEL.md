# DDiF-Inspired Language Modeling in Virgo-2

Virgo-2 includes an experimental language modeling track inspired by DDiF-style "coordinate to quantity" representations.

## How DDiF inspired Virgo-2

Rather than treating text generation only as discrete next-token lookup, Virgo-2 maps character-context coordinates to predicted quantities (e.g., character likelihoods) through a compact field model.

## Coordinate-to-quantity mapping

The LM layer maps context-derived coordinates into output distributions using lightweight neural-field style fitting. Inference samples from these predicted quantities.

## How text is represented

Current implementation is intentionally minimal:

- character-level codec,
- small context windows,
- compact model state designed for experimentation.

## Tiny character LM

CLI commands:

- `virgo2 lm-train <input_txt> <model_dir> --epochs N`
- `virgo2 lm-generate <model_dir> <prompt> --max-chars N`

The tiny LM is intended for fast local iteration and architecture probing.

## Generation loop

1. Encode prompt/context.
2. Query model for next-character distribution.
3. Sample/select next character.
4. Append and repeat until max length.

## Limitations

- Early-stage quality; outputs can be noisy.
- Character-level generation limits fluency.
- Not benchmarked as a production LM.

## Future work

- richer context conditioning,
- improved losses and calibration,
- hybrid symbolic/field decoding,
- evaluation harnesses for perplexity-like diagnostics.
