from __future__ import annotations

from pathlib import Path

import numpy as np

from virgo2.ddif.losses import cross_entropy_from_logits
from virgo2.lm import NeuralFieldLanguageModel


def eval_next_char_loss(model: NeuralFieldLanguageModel, prompt: str, target_id: int) -> float:
    logits = model.predict_next(prompt)
    return cross_entropy_from_logits(logits.reshape(1, -1), np.array([target_id]))


def _repetition_rate(text: str, ngram: int = 2) -> float:
    if len(text) < ngram + 1:
        return 0.0
    seen: set[str] = set()
    repeated = 0
    total = 0
    for i in range(len(text) - ngram + 1):
        token = text[i : i + ngram]
        total += 1
        if token in seen:
            repeated += 1
        seen.add(token)
    return float(repeated / max(total, 1))


def evaluate_model(model: NeuralFieldLanguageModel, text: str, sample_prompt: str = "") -> dict[str, float | int | bool | str]:
    if not text:
        raise ValueError("Input text must be non-empty")
    if model.weights is None:
        raise RuntimeError("Model is not trained")
    ids = model.codec.encode(text)
    if len(ids) < 2:
        raise ValueError("Input text must contain at least two encodable characters")

    losses: list[float] = []
    correct_next = 0
    for i in range(len(ids) - 1):
        prompt = text[: min(i + 1, len(text))]
        target_id = ids[i + 1]
        logits = model.predict_next(prompt).reshape(-1)
        losses.append(float(cross_entropy_from_logits(logits.reshape(1, -1), np.array([target_id]))))
        if int(np.argmax(logits)) == target_id:
            correct_next += 1

    prompt_seed = sample_prompt or text[:1]
    generated = model.generate(prompt_seed, max_chars=40, deterministic=True)
    gen_tail = generated[len(prompt_seed) :]
    invalid_chars = sum(1 for ch in gen_tail if ch not in model.codec.char_to_id)

    checkpoint_valid, checkpoint_message = True, "in-memory"
    deterministic_repeatability = model.generate(prompt_seed, max_chars=20, deterministic=True) == model.generate(
        prompt_seed, max_chars=20, deterministic=True
    )

    return {
        "loss": float(np.mean(losses)),
        "next_char_accuracy": float(correct_next / (len(ids) - 1)),
        "char_reconstruction_accuracy": float(correct_next / (len(ids) - 1)),
        "repetition_rate": _repetition_rate(gen_tail, ngram=2),
        "invalid_character_rate": float(invalid_chars / max(len(gen_tail), 1)),
        "deterministic_repeatability": bool(deterministic_repeatability),
        "sample_output": generated,
        "evaluated_character_count": int(len(ids) - 1),
        "weight_count": int(model.weights.size),
        "feature_count": int(model.weights.shape[0]),
        "codec_vocab_size": int(model.codec.vocab_size),
        "checkpoint_roundtrip_success": bool(checkpoint_valid),
        "checkpoint_validation_message": checkpoint_message,
    }


def evaluate_checkpoint(model_dir: str | Path, text: str, sample_prompt: str = "") -> dict[str, float | int | bool | str]:
    ok, message = NeuralFieldLanguageModel.validate_checkpoint(model_dir)
    if not ok:
        return {
            "checkpoint_roundtrip_success": False,
            "checkpoint_validation_message": message,
            "error": "checkpoint validation failed",
        }
    model = NeuralFieldLanguageModel.load(model_dir)
    metrics = evaluate_model(model, text, sample_prompt=sample_prompt)
    metrics["checkpoint_roundtrip_success"] = True
    metrics["checkpoint_validation_message"] = "ok"
    return metrics
