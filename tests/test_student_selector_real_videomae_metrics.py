from __future__ import annotations

import torch

from training.losses import importance_topk_metrics, pearson_corr_mean


def test_real_videomae_selector_metrics_match_teacher_topk():
    pred_scores = torch.tensor([[0.9, 0.1, 0.8, 0.0], [0.2, 0.95, 0.1, 0.85]])
    target = torch.tensor([[1.0, 0.0, 0.7, 0.1], [0.0, 1.0, 0.2, 0.9]])

    metrics = importance_topk_metrics(pred_scores, target, k=2)

    assert metrics["importance_mse"] >= 0.0
    assert metrics["importance_mae"] >= 0.0
    assert metrics["pearson_corr_mean"] > 0.0
    assert metrics["target_top1_overlap"] == 1.0
    assert metrics["target_topk_overlap"] == 1.0
    assert metrics["selected_teacher_importance_mean"] > metrics["random_teacher_importance_mean"]


def test_real_videomae_selector_constant_correlation_is_zero_not_nan():
    pred_scores = torch.ones(2, 4)
    target = torch.ones(2, 4)

    corr = pearson_corr_mean(pred_scores, target)
    metrics = importance_topk_metrics(pred_scores, target, k=2)

    assert torch.isfinite(corr)
    assert corr.item() == 0.0
    assert metrics["pearson_corr_mean"] == 0.0
