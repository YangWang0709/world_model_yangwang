from __future__ import annotations

import math

import torch

from eval.eval_efficiency import token_retention_ratio
from eval.eval_student_world_model import future_mse_value
from eval.eval_teacher_student_gap import compute_teacher_student_gap
from training.student_world_model_trainer import selection_quality_metrics


def test_real_videomae_student_world_model_metrics_are_finite() -> None:
    pred_future = torch.tensor([[0.0, 1.0, 2.0], [1.0, 1.5, 2.5]])
    target_future = torch.tensor([[0.0, 0.5, 2.5], [1.5, 1.5, 2.0]])
    teacher_future_mse = 0.125
    student_future_mse = future_mse_value(pred_future, target_future)

    gap = compute_teacher_student_gap(
        teacher_future_mse=teacher_future_mse,
        student_future_mse=student_future_mse,
    )

    assert student_future_mse >= 0.0
    assert math.isfinite(gap["student_teacher_ratio"])
    assert gap["student_teacher_gap"] == student_future_mse - teacher_future_mse


def test_real_videomae_no_key_mask_selection_metrics_use_teacher_importance() -> None:
    scores = torch.tensor([[0.9, 0.2, 0.8, 0.1], [0.1, 0.7, 0.2, 0.8]])
    target_importance = torch.tensor([[1.0, 0.0, 0.6, 0.2], [0.1, 0.7, 0.0, 1.0]])

    metrics = selection_quality_metrics(scores, key_token_mask=None, target_importance=target_importance, k=2)

    assert metrics["selector_target_top1_overlap"] == 1.0
    assert metrics["selector_target_topk_overlap"] == 1.0
    assert metrics["selected_teacher_importance_mean"] > metrics["random_teacher_importance_mean"]
    assert math.isfinite(metrics["selected_vs_random_importance_gap"])
    assert token_retention_ratio(selected_tokens=2, total_tokens=4) == 0.5
