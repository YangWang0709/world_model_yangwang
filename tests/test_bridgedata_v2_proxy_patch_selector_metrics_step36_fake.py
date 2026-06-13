import torch

from data.bridgedata_v2_proxy_patch_selector_metrics_step36 import (
    build_step36_gate_decision,
    combined_patch_selector_loss,
    compute_patch_prediction_metrics,
)


def test_step36_metrics_compare_patch_selector_against_token_baselines():
    target = torch.zeros(2, 4, 5)
    target[:, 0, 0] = 1.0
    target[:, 1, 2] = 0.75
    target[:, 2, 4] = 0.5
    pred = target.clone()
    metrics = compute_patch_prediction_metrics(pred, target, random_seed=1, topk_values=[2, 4, 8])
    assert metrics["patch_selector_val_mse"] == 0.0
    assert metrics["patch_selector_beats_random_token_baseline"] is True
    assert metrics["patch_selector_beats_uniform_token_baseline"] is True
    assert metrics["patch_selector_beats_temporal_broadcast_baseline"] is True
    assert metrics["patch_selector_beats_current_only_patch_baseline"] is True
    assert metrics["top2_overlap"] == 1.0
    assert metrics["top4_overlap"] == 1.0
    assert metrics["top256_precision"] == 1.0
    assert metrics["patch_selector_pearson"] > 0.99
    assert metrics["patch_selector_spearman"] > 0.99


def test_step36_combined_loss_is_finite_and_backwardable():
    logits = torch.randn(2, 4, 5, requires_grad=True)
    target = torch.rand(2, 4, 5)
    parts = combined_patch_selector_loss(
        logits,
        target,
        {
            "mse_weight": 1.0,
            "rank_weight": 0.1,
            "topk_soft_weight": 0.1,
            "rank_pairs_per_batch": 16,
            "topk_values": [2, 4],
        },
    )
    parts["loss"].backward()
    assert torch.isfinite(parts["loss"]).item()
    assert logits.grad is not None


def test_step36_gate_never_allows_final_selector_training():
    good = {
        "num_rows": 1,
        "patch_selector_beats_random_token_baseline": True,
        "patch_selector_beats_uniform_token_baseline": True,
        "patch_selector_beats_temporal_broadcast_baseline": True,
        "top256_overlap": 0.2,
    }
    gate = build_step36_gate_decision(
        within=good,
        cross=good,
        mixed=good,
        leakage={
            "no_language_or_trajectory_leakage": True,
            "train_val_sample_id_disjoint": True,
            "train_val_trajectory_disjoint_or_documented": True,
        },
        dataset_bias_detected=False,
        full_context_noise_acknowledged=True,
        top256_overlap_mean_min=0.10,
    )
    assert gate["future_patch_selector_gate_ready"] is True
    assert gate["selector_training_allowed"] is False
    assert gate["final_selector_training_allowed"] is False
    assert gate["current_importance_training_allowed"] is False
    assert gate["context_utility_claim_allowed"] is False
