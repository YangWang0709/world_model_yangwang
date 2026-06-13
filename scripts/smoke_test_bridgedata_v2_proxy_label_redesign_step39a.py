"""Fake-tensor smoke gate for Step39A proxy-label redesign."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_proxy_label_redesign_step39a import (
    STAGE,
    aggregate_variant_metrics,
    build_coarse_grid_label_if_possible,
    build_coarse_index_bins_label,
    build_denoised_soft_topk_label,
    build_global_spatial_prior,
    build_global_spatial_prior_removed_residual_label,
    build_label_variants_for_sample,
    build_step39a_gate_decision,
    build_temporal_broadcast_label,
    build_temporal_only_label,
    compute_cross_shard_variant_consistency,
    compute_label_variant_metrics,
    rank_label_variants,
)


def main() -> None:
    torch.manual_seed(42)
    base = torch.rand(16, 392)
    temporal = build_temporal_only_label(base)
    assert list(temporal["native_label"].shape) == [16]
    assert list(temporal["broadcast_label"].shape) == [16, 392]
    assert list(build_temporal_broadcast_label(base).shape) == [16, 392]
    for bins in (49, 98, 196):
        assert list(build_coarse_index_bins_label(base, bins).shape) == [16, 392]
    assert list(build_coarse_grid_label_if_possible(base, [14, 28], [7, 14]).shape) == [16, 392]
    soft_topk = build_denoised_soft_topk_label(base, 256, 0.15, 0.02)
    assert list(soft_topk.shape) == [16, 392]
    assert bool(torch.isfinite(soft_topk).all())
    assert float(soft_topk.min()) >= 0.0 and float(soft_topk.max()) <= 1.0
    samples = [
        {"importance": base, "shard_id": "shard0"},
        {"importance": base * 0.8 + 0.1 * torch.rand(16, 392), "shard_id": "shard1"},
    ]
    prior = build_global_spatial_prior(samples)["prior"]
    assert list(build_global_spatial_prior_removed_residual_label(base, prior).shape) == [16, 392]
    config = {
        "label_variants": {
            "enabled": [
                "temporal_only",
                "temporal_broadcast",
                "coarse_index_bins_49",
                "coarse_index_bins_98",
                "coarse_index_bins_196",
                "coarse_grid_7x14_if_14x28",
                "denoised_soft_topk_256",
                "global_spatial_prior_removed_residual",
            ],
            "topk_values": [64, 128, 256, 512],
            "coarse_index_bins": [49, 98, 196],
            "optional_grid_shape": [14, 28],
            "coarse_grid_shapes": [[7, 14]],
            "soft_topk_temperature": 0.15,
            "soft_topk_floor": 0.02,
        }
    }
    rows = []
    for idx, sample in enumerate(samples):
        variants = build_label_variants_for_sample(sample["importance"], prior, config)
        for name, label in variants.items():
            row = compute_label_variant_metrics(sample["importance"], label, name, [64, 128, 256, 512])
            row.update({"variant_name": name, "shard_id": sample["shard_id"], "_label_flat": label.reshape(-1)})
            rows.append(row)
    cross = compute_cross_shard_variant_consistency(rows)
    aggregates = {name: aggregate_variant_metrics(rows, name) for name in sorted({row["variant_name"] for row in rows})}
    comparison = {
        "stage": STAGE,
        "ranking": rank_label_variants(
            {"aggregates": aggregates, "cross_shard_consistency": cross},
            {
                "primary_topk_mass_min": 0.25,
                "entropy_max_for_patch_like_label": 0.85,
                "cross_shard_consistency_min": 0.70,
                "reconstruction_mse_reasonable_max": 0.08,
                "residual_energy_after_redesign_max": 0.35,
            },
        ),
    }
    assert comparison["ranking"]["best_variant"]
    gate = build_step39a_gate_decision(comparison=comparison, step38b_gate={"spatial_residual_learnable_evidence": False})
    for flag in (
        "selector_training_allowed",
        "final_selector_training_allowed",
        "current_importance_training_allowed",
        "context_utility_claim_allowed",
        "optimizer_step_performed",
        "checkpoint_saved",
        "state_dict_saved",
    ):
        assert bool(gate.get(flag, False)) is False
    print(
        json.dumps(
            {
                "BRIDGEDATA_V2_TFDS_PROXY_LABEL_REDESIGN_STEP39A_PASS": True,
                "best_variant": comparison["ranking"]["best_variant"]["variant_name"],
                "recommended_step40_label_variant": gate["recommended_step40_label_variant"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
