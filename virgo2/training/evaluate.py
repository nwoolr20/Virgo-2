from __future__ import annotations

from virgo2.ddif.losses import cross_entropy_from_logits
from virgo2.lm import NeuralFieldLanguageModel


def eval_next_char_loss(model: NeuralFieldLanguageModel, prompt: str, target_id: int) -> float:
    logits = model.predict_next(prompt)
    return cross_entropy_from_logits(logits.reshape(1, -1), __import__('numpy').array([target_id]))
