from __future__ import annotations

import torch

from eval.eval_importance_quality import compute_quality_metrics
from scripts.generate_predictive_importance import normalize_importance_scores


def test_importance_quality_prefers_key_tokens() -> None:
    scores = torch.tensor(
        [
            [0.1, 3.0, 0.2, 2.0],
            [4.0, 0.0, 3.0, 0.2],
        ]
    )
    key_mask = torch.tensor(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=torch.float32,
    )

    metrics = compute_quality_metrics(scores, key_mask)

    assert metrics["key_token_importance_mean"] > metrics["non_key_token_importance_mean"]
    assert metrics["top1_hit_rate"] == 1.0
    assert metrics["topk_hit_rate"] == 1.0


def test_minmax_normalization_constant_rows_are_finite_zero() -> None:
    normalized = normalize_importance_scores(torch.ones(2, 4), mode="minmax_per_sample")

    assert torch.isfinite(normalized).all()
    assert torch.equal(normalized, torch.zeros(2, 4))
