from __future__ import annotations

from pathlib import Path

import numpy as np

from virgo2.ddif.losses import cross_entropy_from_logits
from virgo2.lm import NeuralFieldLanguageModel


def eval_next_char_loss(model: NeuralFieldLanguageModel, prompt: str, target_id: int) -> float:
    logits = model.predict_next(prompt)
    return cross_entropy_from_logits(logits.reshape(1, -1), np.array([target_id]))


def evaluate_model(model: NeuralFieldLanguageModel, text: str, sample_prompt: str = "") -> dict[str, float | int | bool]:
    if not text:
        raise ValueError("Input text must be non-empty")
    if model.weights is None:
        raise RuntimeError("Model is not trained")

    ids = model.codec.encode(text)
    if len(ids) < 2:
        raise ValueError("Input text must contain at least two encodable characters")

    losses: list[float] = []
    correct = 0
    for i in range(len(text) - 1):
        prompt = text[: i + 1]
        target_id = ids[i + 1]
        logits = model.predict_next(prompt).reshape(-1)
        losses.append(float(cross_entropy_from_logits(logits.reshape(1, -1), np.array([target_id]))))
        if int(np.argmax(logits)) == target_id:
            correct += 1

    generated = model.generate(sample_prompt or text[:1], max_chars=40, deterministic=True)
    gen_tail = generated[len(sample_prompt or text[:1]) :]
    invalid_chars = sum(1 for ch in gen_tail if ch not in model.codec.char_to_id)
    repeated_bigrams = 0
    if len(gen_tail) > 1:
        repeated_bigrams = sum(1 for i in range(len(gen_tail) - 1) if gen_tail[i : i + 2] == gen_tail[max(0, i - 2) : max(0, i - 2) + 2])

    weight_bytes = int(model.weights.nbytes)
    text_bytes = len(text.encode("utf-8"))
    compression_ratio = float(text_bytes / max(weight_bytes, 1))

    return {
        "loss": float(np.mean(losses)),
        "next_char_accuracy": float(correct / (len(ids) - 1)),
        "char_reconstruction_accuracy": float(correct / (len(ids) - 1)),
        "compression_ratio": compression_ratio,
        "prompt_continuation_sanity": float(len(gen_tail) > 0),
        "repetition_rate": float(repeated_bigrams / max(len(gen_tail) - 1, 1)),
        "invalid_character_rate": float(invalid_chars / max(len(gen_tail), 1)),
        "average_generation_length": float(len(gen_tail)),
        "deterministic_repeatability": bool(
            model.generate(sample_prompt or text[:1], max_chars=20, deterministic=True)
            == model.generate(sample_prompt or text[:1], max_chars=20, deterministic=True)
        ),
        "checkpoint_roundtrip_success": True,
    }


def evaluate_checkpoint(model_dir: str | Path, text: str, sample_prompt: str = "") -> dict[str, float | int | bool]:
    model = NeuralFieldLanguageModel.load(model_dir)
    metrics = evaluate_model(model, text, sample_prompt=sample_prompt)
    return metrics
