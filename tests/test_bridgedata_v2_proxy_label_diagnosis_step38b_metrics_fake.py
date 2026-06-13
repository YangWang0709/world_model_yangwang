import torch

from data.bridgedata_v2_proxy_label_diagnosis_step38b import (
    build_cross_shard_consistency,
    build_step38b_gate_decision,
    compute_label_entropy,
    compute_temporal_broadcast_fit,
    compute_topk_concentration,
    decompose_proxy_importance,
)


def test_step38b_entropy_topk_and_fit_are_finite():
    torch.manual_seed(42)
    importance = torch.rand(16, 392)
    decomposition = decompose_proxy_importance(importance)

    entropy = compute_label_entropy(importance)
    topk = compute_topk_concentration(importance, [64, 128, 256, 512])
    fit = compute_temporal_broadcast_fit(importance, decomposition["temporal_broadcast"])

    for payload in (entropy, topk, fit):
        for value in payload.values():
            assert torch.isfinite(torch.tensor(float(value)))
    assert 0.0 <= entropy["global_normalized_entropy"] <= 1.0
    assert 0.0 <= topk["top256_mass_ratio"] <= 1.0
    assert 0.0 <= topk["top256_frame_coverage"] <= 1.0


def test_step38b_gate_never_enables_training_flags():
    summary = {
        "stage": "bridgedata_v2_tfds_proxy_label_diagnosis_step38b",
        "safe_stop": False,
        "diagnosis_performed": True,
        "all_samples": {
            "residual_energy_ratio_mean": 0.8,
            "per_frame_spatial_entropy_mean_mean": 0.1,
            "top256_mass_ratio_mean": 0.8,
            "temporal_broadcast_r2_like_mean": 0.1,
        },
    }
    cross = {"cross_shard_residual_consistent": True}

    gate = build_step38b_gate_decision(
        config={"diagnostics": {"primary_topk": 256, "thresholds": {}}},
        diagnosis_summary=summary,
        cross_shard_consistency=cross,
        leakage_summary={"no_language_or_trajectory_leakage": True},
    )

    assert gate["spatial_residual_learnable_evidence"] is True
    assert gate["selector_training_allowed"] is False
    assert gate["final_selector_training_allowed"] is False
    assert gate["current_importance_training_allowed"] is False
    assert gate["context_utility_claim_allowed"] is False
    assert gate["label_patch_detail_learnable_allowed"] is False


def test_step38b_cross_shard_consistency_returns_finite_similarity():
    shared = torch.rand(16, 392)
    payload = build_cross_shard_consistency(
        [
            {"shard_id": "shard0", "importance": shared},
            {"shard_id": "shard1", "importance": shared + 0.01 * torch.rand(16, 392)},
        ]
    )

    assert payload["computed"] is True
    assert torch.isfinite(torch.tensor(float(payload["cross_shard_residual_cosine"])))
