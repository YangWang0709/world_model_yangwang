import torch

from data.bridgedata_v2_proxy_label_redesign_step39a import (
    aggregate_variant_metrics,
    build_step39a_gate_decision,
    compute_cross_shard_variant_consistency,
    compute_label_variant_metrics,
    rank_label_variants,
)


def test_step39a_metrics_ranking_and_gate_are_finite_and_safe():
    original = torch.rand(16, 392)
    variant = original.mean(dim=1, keepdim=True).expand_as(original)
    row0 = compute_label_variant_metrics(original, variant, "temporal_broadcast", [64, 128, 256, 512])
    row1 = compute_label_variant_metrics(original * 0.9, variant * 0.9, "temporal_broadcast", [64, 128, 256, 512])
    row0.update({"shard_id": "shard0", "_label_flat": variant.reshape(-1)})
    row1.update({"shard_id": "shard1", "_label_flat": (variant * 0.9).reshape(-1)})
    rows = [row0, row1]
    aggregate = aggregate_variant_metrics(rows, "temporal_broadcast")
    cross = compute_cross_shard_variant_consistency(rows)
    ranking = rank_label_variants(
        {"aggregates": {"temporal_broadcast": aggregate}, "cross_shard_consistency": cross},
        {
            "primary_topk_mass_min": 0.25,
            "entropy_max_for_patch_like_label": 0.85,
            "reconstruction_mse_reasonable_max": 0.08,
            "residual_energy_after_redesign_max": 0.35,
        },
    )
    gate = build_step39a_gate_decision(
        comparison={"ranking": ranking},
        step38b_gate={"spatial_residual_learnable_evidence": False},
    )

    assert torch.isfinite(torch.tensor(float(aggregate["reconstruction_mse_to_original_mean"])))
    assert cross["by_variant"]["temporal_broadcast"]["computed"] is True
    assert ranking["best_variant"]["variant_name"] == "temporal_broadcast"
    assert gate["selector_training_allowed"] is False
    assert gate["final_selector_training_allowed"] is False
    assert gate["current_importance_training_allowed"] is False
    assert gate["context_utility_claim_allowed"] is False
