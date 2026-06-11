from __future__ import annotations

import math

import torch

from training.losses import importance_topk_metrics, pearson_corr_mean


def test_bair_selector_importance_metrics_are_finite_and_interpretable():
    target = torch.tensor(
        [
            [0.0, 0.1, 0.8, 1.0, 0.2, 0.4],
            [0.9, 0.1, 0.2, 0.3, 0.7, 0.0],
        ],
        dtype=torch.float32,
    )
    pred = torch.tensor(
        [
            [0.05, 0.1, 0.7, 0.95, 0.3, 0.2],
            [0.8, 0.2, 0.25, 0.1, 0.65, 0.05],
        ],
        dtype=torch.float32,
    )

    metrics = importance_topk_metrics(pred, target, k=2, random_seed=123)

    for key in (
        "importance_mse",
        "importance_mae",
        "pearson_corr_mean",
        "target_top1_overlap",
        "target_topk_overlap",
        "selected_teacher_importance_mean",
        "random_teacher_importance_mean",
        "selected_vs_random_importance_gap",
        "score_mean",
        "target_mean",
    ):
        assert math.isfinite(float(metrics[key]))
    assert metrics["target_top1_overlap"] == 1.0
    assert metrics["target_topk_overlap"] == 1.0
    assert metrics["selected_vs_random_importance_gap"] == (
        metrics["selected_teacher_importance_mean"] - metrics["random_teacher_importance_mean"]
    )


def test_bair_selector_pearson_constant_rows_return_zero_not_nan():
    pred = torch.ones(2, 6)
    target = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    corr = pearson_corr_mean(pred, target)

    assert torch.isfinite(corr)
    assert float(corr.item()) == 0.0
