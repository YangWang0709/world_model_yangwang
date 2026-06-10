from __future__ import annotations

import torch

from eval.eval_student_world_model import future_mse_value
from training.student_world_model_trainer import selected_key_metrics


def test_future_mse_value_matches_expected_scalar() -> None:
    pred = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 4.0]])

    assert future_mse_value(pred, target) == 2.0


def test_selected_key_metrics_tracks_topk_coverage() -> None:
    scores = torch.tensor(
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

    metrics = selected_key_metrics(scores, key_mask, k=2)

    assert metrics["top1_hit_rate"] == 1.0
    assert metrics["topk_hit_rate"] == 1.0
    assert metrics["selected_key_coverage"] == 1.0
    assert metrics["selected_key_fraction"] == 1.0
