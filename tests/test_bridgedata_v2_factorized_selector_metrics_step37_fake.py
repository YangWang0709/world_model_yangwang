import torch

from data.bridgedata_v2_factorized_selector_metrics_step37 import (
    build_step37_gate_decision,
    compute_factorized_prediction_metrics,
)


def test_step37_metrics_compare_factorized_against_baselines_and_step36():
    target = torch.zeros(2, 4, 5)
    target[:, 0, 0] = 1.0
    target[:, 1, 2] = 0.75
    target[:, 2, 4] = 0.5
    pred = target.clone()
    metrics = compute_factorized_prediction_metrics(
        pred,
        target,
        random_seed=1,
        topk_values=[2, 4, 8],
        step36_reference_mse=0.1,
    )
    assert metrics["factorized_selector_val_mse"] == 0.0
    assert metrics["factorized_beats_random"] is True
    assert metrics["factorized_beats_uniform"] is True
    assert metrics["factorized_beats_temporal_broadcast"] is True
    assert metrics["factorized_beats_current_only"] is True
    assert metrics["factorized_beats_step36_direct_patch"] is True
    assert metrics["top2_overlap"] == 1.0
    assert metrics["top256_precision"] == 1.0
    assert metrics["pearson"] > 0.99
    assert metrics["spearman"] > 0.99


def test_step37_gate_never_allows_final_selector_training():
    good = {
        "num_rows": 1,
        "factorized_beats_random": True,
        "factorized_beats_uniform": True,
        "factorized_beats_temporal_broadcast": True,
        "factorized_beats_current_only": True,
        "factorized_beats_step36_direct_patch": True,
        "top256_overlap": 0.2,
    }
    gate = build_step37_gate_decision(
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
        top256_overlap_mean_min=0.12,
    )
    assert gate["future_factorized_selector_gate_ready"] is True
    assert gate["selector_training_allowed"] is False
    assert gate["final_selector_training_allowed"] is False
    assert gate["current_importance_training_allowed"] is False
    assert gate["context_utility_claim_allowed"] is False
