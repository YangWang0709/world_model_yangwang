from __future__ import annotations

import torch

from training.losses import topk_coverage_metrics


def test_student_selector_topk_metrics_high_when_key_scores_high() -> None:
    pred_scores = torch.tensor(
        [
            [0.9, 0.1, 0.8, 0.0],
            [0.2, 0.95, 0.1, 0.85],
        ]
    )
    key_mask = torch.tensor(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ],
        dtype=torch.float32,
    )

    metrics = topk_coverage_metrics(pred_scores, key_mask, k=2)

    assert metrics["key_score_mean"] > metrics["non_key_score_mean"]
    assert metrics["top1_hit_rate"] == 1.0
    assert metrics["topk_hit_rate"] == 1.0
