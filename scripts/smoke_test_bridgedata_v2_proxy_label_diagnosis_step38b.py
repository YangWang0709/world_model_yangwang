"""Fake-tensor smoke gate for Step38B proxy label diagnosis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_proxy_label_diagnosis_step38b import (
    STAGE,
    aggregate_label_diagnosis,
    build_cross_shard_consistency,
    build_step38b_gate_decision,
    compute_label_entropy,
    compute_temporal_broadcast_fit,
    compute_topk_concentration,
    decompose_proxy_importance,
    diagnose_step38b_sample,
)


def main() -> None:
    torch.manual_seed(42)
    temporal = torch.linspace(0.0, 1.0, 16, dtype=torch.float32)[:, None].expand(16, 392).contiguous()
    noise = torch.rand(16, 392, dtype=torch.float32)
    temporal_decomp = decompose_proxy_importance(temporal)
    noise_decomp = decompose_proxy_importance(noise)
    entropy = compute_label_entropy(noise)
    topk = compute_topk_concentration(noise, [64, 128, 256, 512])
    fit = compute_temporal_broadcast_fit(temporal, temporal_decomp["temporal_broadcast"])
    if list(temporal_decomp["temporal_component"].shape) != [16]:
        raise AssertionError("bad temporal_component shape")
    if list(temporal_decomp["temporal_broadcast"].shape) != [16, 392]:
        raise AssertionError("bad temporal_broadcast shape")
    if list(temporal_decomp["spatial_residual"].shape) != [16, 392]:
        raise AssertionError("bad spatial_residual shape")
    if float(temporal_decomp["residual_energy_ratio"]) > 1.0e-8:
        raise AssertionError("temporal-only label residual should be near zero")
    if float(noise_decomp["residual_energy_ratio"]) <= 0.05:
        raise AssertionError("random patch noise should have visible residual energy")
    _assert_finite_dict(entropy)
    _assert_finite_dict(topk)
    _assert_finite_dict(fit)
    rows = [
        diagnose_step38b_sample(
            {
                "sample_id": "fake0",
                "trajectory_id": "traj0",
                "shard_id": "shard0",
                "data_package_id": "fake",
                "importance": temporal,
            },
            [64, 128, 256, 512],
        ),
        diagnose_step38b_sample(
            {
                "sample_id": "fake1",
                "trajectory_id": "traj1",
                "shard_id": "shard1",
                "data_package_id": "fake",
                "importance": noise,
            },
            [64, 128, 256, 512],
        ),
    ]
    summary = {
        "stage": STAGE,
        "safe_stop": False,
        "diagnosis_performed": True,
        "all_samples": aggregate_label_diagnosis(rows, "all_samples", primary_topk=256),
    }
    cross = build_cross_shard_consistency(
        [
            {"shard_id": "shard0", "importance": temporal},
            {"shard_id": "shard1", "importance": noise},
        ]
    )
    gate = build_step38b_gate_decision(
        config={
            "diagnostics": {"primary_topk": 256, "thresholds": {}},
            "future_gates": {
                "selector_training_allowed": False,
                "final_selector_training_allowed": False,
                "current_importance_training_allowed": False,
                "context_utility_claim_allowed": False,
            },
        },
        diagnosis_summary=summary,
        cross_shard_consistency=cross,
        leakage_summary={"no_language_or_trajectory_leakage": True},
    )
    for flag in (
        "selector_training_allowed",
        "final_selector_training_allowed",
        "current_importance_training_allowed",
        "context_utility_claim_allowed",
        "label_patch_detail_learnable_allowed",
        "optimizer_step_performed",
        "checkpoint_saved",
        "state_dict_saved",
    ):
        if bool(gate.get(flag, False)):
            raise AssertionError(f"{flag} must remain false")
    print(
        json.dumps(
            {
                "BRIDGEDATA_V2_TFDS_PROXY_LABEL_DIAGNOSIS_STEP38B_PASS": True,
                "residual_energy_temporal_only": temporal_decomp["residual_energy_ratio"],
                "residual_energy_random_noise": noise_decomp["residual_energy_ratio"],
                "recommended_step39": gate["recommended_step39"]["name"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _assert_finite_dict(payload: dict[str, object]) -> None:
    for key, value in payload.items():
        if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
            raise AssertionError(f"{key} is not finite")


if __name__ == "__main__":
    main()
