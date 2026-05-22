# Virgo-2 Training Roadmap

## Current Stage

Virgo-2 LM training is currently deterministic closed-form neural-field fitting via ridge regression.
This path is non-iterative and enforces `epochs == 1` as a compatibility contract.

## Future Possible Stages

The following are roadmap concepts only (not implemented in current Virgo-2):

- iterative optimization,
- mini-batch fitting,
- streaming field adaptation,
- local incremental updates,
- hybrid latent-field optimization.

## Contract Integrity Note

Until a true iterative optimizer is introduced, docs, tests, and CLI examples must continue to treat multi-epoch training as unsupported.
