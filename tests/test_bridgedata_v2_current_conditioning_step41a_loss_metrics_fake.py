import torch

from data.bridgedata_v2_current_conditioning_losses_step41a import current_conditioning_mse_loss
from data.bridgedata_v2_current_conditioning_metrics_step41a import (
    add_current_gain_metrics,
    build_step41a_gate_decision,
    build_variant_comparison,
    compute_current_conditioning_metrics,
)


def test_step41a_loss_and_metrics_are_finite_for_fake_batch():
    logits = torch.randn(2, 4, 5)
    target = torch.rand(2, 4, 5)
    prior = torch.rand(5)

    loss = current_conditioning_mse_loss({"scores": logits}, target, {"use_sigmoid_scores": True})
    metrics = compute_current_conditioning_metrics(torch.sigmoid(logits), target, prior, topk_values=[2, 4])

    assert torch.isfinite(loss["loss"])
    assert metrics["all_finite"] is True
    assert "random_token_baseline_mse" in metrics
    assert "uniform_or_mean_baseline_mse" in metrics
    assert "temporal_broadcast_baseline_mse" in metrics
    assert "train_global_spatial_prior_baseline_mse" in metrics


def test_step41a_variant_comparison_and_gate_keep_final_flags_false():
    rows = [
        _row("same", "within_shard", "no_current_context_only", 0.30),
        _row("same", "within_shard", "current_mean_summary", 0.20),
        _row("same", "within_shard", "current_coarse_spatial_query_attention", 0.10),
        _row("cross", "cross_shard", "no_current_context_only", 0.30),
        _row("cross", "cross_shard", "current_mean_summary", 0.20),
        _row("cross", "cross_shard", "current_coarse_spatial_query_attention", 0.10),
        _row("mixed", "mixed_shard", "no_current_context_only", 0.30),
        _row("mixed", "mixed_shard", "current_mean_summary", 0.20),
        _row("mixed", "mixed_shard", "current_coarse_spatial_query_attention", 0.10),
    ]
    add_current_gain_metrics(rows)
    comparison = build_variant_comparison(
        rows,
        ["no_current_context_only", "current_mean_summary", "current_coarse_spatial_query_attention"],
    )
    gate = build_step41a_gate_decision(
        variant_comparison=comparison,
        leakage={"no_language_or_trajectory_leakage": True},
        requirements={"top256_overlap_mean_min": 0.12, "overfit_gap_max": 0.05},
    )

    assert comparison["best_variant"] == "current_coarse_spatial_query_attention"
    assert gate["current_conditioning_candidate_ready"] is True
    assert gate["best_variant_beats_no_current"] is True
    assert gate["best_variant_beats_current_mean_summary"] is True
    assert gate["downstream_selector_use_allowed"] is False
    assert gate["final_selector_training_allowed"] is False
    assert gate["current_importance_training_allowed"] is False
    assert gate["context_utility_claim_allowed"] is False


def _row(setting: str, eval_type: str, variant: str, mse: float):
    return {
        "setting": setting,
        "eval_type": eval_type,
        "variant_name": variant,
        "selector_val_mse": mse,
        "selector_val_mae": mse,
        "random_token_baseline_mse": 0.9,
        "uniform_or_mean_baseline_mse": 0.8,
        "temporal_broadcast_baseline_mse": 0.7,
        "train_global_spatial_prior_baseline_mse": 0.6,
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
        "train_mse": mse - 0.01,
        "overfit_gap": 0.01,
        "all_finite": True,
    }
