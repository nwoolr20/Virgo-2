# Virgo-2

Virgo-2 is a lightweight neural-field memory and language-model research system with three layers:
1. Core neural-field memory substrate.
2. Continuous field lifecycle + conversational memory.
3. Experimental DDiF-inspired neural-field LM.

This is **not GPT-class**, not production AGI, and generation remains experimental.
Field folding is deterministic/simple. Memory is early-stage but functional.

## Install
```bash
python -m pip install -e ".[dev]"
```

## Lifecycle quickstart
```bash
virgo2 vault-init ./vault
virgo2 remember ./vault "Remember that Virgo-2 is a neural-field LM."
virgo2 recall ./vault "What is Virgo-2?"
virgo2 chat-memory ./vault "Do you remember what I am building?"
virgo2 fold ./vault conversation_core folded_conversation_0001
virgo2 forge-check ./vault
```
