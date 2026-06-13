import torch

from data.bridgedata_v2_proxy_temporal_selector_metrics_step35 import (
    build_step35_gate_decision,
    combined_temporal_selector_loss,
    compute_temporal_prediction_metrics,
)


def test_step35_metrics_compare_selector_against_temporal_baselines():
    target = torch.tensor([[0.0, 1.0, 0.5, 0.25], [1.0, 0.0, 0.25, 0.5]])
    pred = target.clone()
    metrics = compute_temporal_prediction_metrics(pred, target, random_seed=1)
    assert metrics["selector_val_mse"] == 0.0
    assert metrics["selector_beats_random_baseline"] is True
    assert metrics["selector_beats_current_only_baseline"] is True
    assert metrics["selector_beats_uniform_baseline"] is True
    assert metrics["top1_frame_hit"] == 1.0
    assert metrics["top2_frame_overlap"] == 1.0
    assert metrics["top4_frame_overlap"] == 1.0
    assert metrics["selector_pearson"] > 0.99
    assert metrics["selector_spearman"] > 0.99


def test_step35_combined_loss_is_finite_and_backwardable():
    logits = torch.randn(2, 4, requires_grad=True)
    target = torch.rand(2, 4)
    parts = combined_temporal_selector_loss(
        logits,
        target,
        {"mse_weight": 1.0, "rank_weight": 0.1, "topk_soft_weight": 0.1, "topk_frames": [1, 2]},
    )
    parts["loss"].backward()
    assert torch.isfinite(parts["loss"]).item()
    assert logits.grad is not None


def test_step35_gate_never_allows_final_selector_training():
    good = {
        "num_rows": 1,
        "selector_beats_random_baseline": True,
        "selector_beats_current_only_baseline": True,
    }
    gate = build_step35_gate_decision(
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
    )
    assert gate["future_selector_training_gate_ready"] is True
    assert gate["selector_training_allowed"] is False
    assert gate["final_selector_training_allowed"] is False
    assert gate["current_importance_training_allowed"] is False
    assert gate["context_utility_claim_allowed"] is False
