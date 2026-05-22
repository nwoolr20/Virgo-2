from __future__ import annotations


def estimate_param_budget(input_dim: int, hidden_dim: int, output_dim: int) -> int:
    return input_dim * hidden_dim + hidden_dim * output_dim + hidden_dim + output_dim
