import torch

from data.bridgedata_v2_redesigned_label_selector_losses_step40a import redesigned_label_selector_mse_loss
from data.bridgedata_v2_redesigned_label_selector_metrics_step40a import (
    aggregate_redesigned_label_selector_metrics,
    build_step40a_gate_decision,
    compute_redesigned_label_selector_metrics,
)


def test_step40a_loss_and_metrics_are_finite_and_safe():
    target = torch.rand(3, 4, 5)
    logits = torch.randn(3, 4, 5)
    loss = redesigned_label_selector_mse_loss(logits, target, {"use_sigmoid_scores": True})
    metrics = compute_redesigned_label_selector_metrics(torch.sigmoid(logits), target, torch.rand(5), topk_values=[64, 128, 256, 512])

    assert torch.isfinite(loss["loss"])
    assert metrics["all_finite"] is True
    assert torch.isfinite(torch.tensor(float(metrics["selector_val_mse"])))
    assert "selector_beats_random_token_baseline" in metrics
    assert "selector_beats_train_global_spatial_prior_baseline" in metrics


def test_step40a_gate_keeps_downstream_and_final_flags_false_even_when_passes():
    row = {
        "selector_val_mse": 0.1,
        "selector_val_mae": 0.1,
        "random_token_baseline_mse": 0.2,
        "uniform_or_mean_baseline_mse": 0.2,
        "temporal_broadcast_baseline_mse": 0.2,
        "train_global_spatial_prior_baseline_mse": 0.2,
        "selector_beats_random_token_baseline": True,
        "selector_beats_uniform_or_mean_baseline": True,
        "selector_beats_temporal_broadcast_baseline": True,
        "selector_beats_train_global_spatial_prior_baseline": True,
        "pearson": 0.1,
        "spearman": 0.1,
        "top64_overlap": 0.2,
        "top128_overlap": 0.2,
        "top256_overlap": 0.2,
        "top512_overlap": 0.2,
        "top256_precision": 0.2,
        "top256_recall": 0.2,
        "train_mse": 0.09,
        "overfit_gap": 0.01,
        "all_finite": True,
    }
    within = aggregate_redesigned_label_selector_metrics([row], eval_type="within_shard")
    cross = aggregate_redesigned_label_selector_metrics([row], eval_type="cross_shard")
    mixed = aggregate_redesigned_label_selector_metrics([row], eval_type="mixed_shard")

    gate = build_step40a_gate_decision(
        within=within,
        cross=cross,
        mixed=mixed,
        leakage={"no_language_or_trajectory_leakage": True},
        requirements={"top256_overlap_mean_min": 0.12, "overfit_gap_max": 0.05},
    )

    assert gate["redesigned_label_selector_smoke_pass"] is True
    assert gate["downstream_selector_use_allowed"] is False
    assert gate["final_selector_training_allowed"] is False
    assert gate["current_importance_training_allowed"] is False
    assert gate["context_utility_claim_allowed"] is False
