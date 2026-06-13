import torch

from data.bridgedata_v2_proxy_label_diagnosis_step38b import (
    aggregate_label_diagnosis,
    decompose_proxy_importance,
    diagnose_step38b_sample,
)


def test_step38b_decomposes_fake_importance_shapes():
    importance = torch.rand(16, 392)

    payload = decompose_proxy_importance(importance)

    assert list(payload["temporal_component"].shape) == [16]
    assert list(payload["temporal_broadcast"].shape) == [16, 392]
    assert list(payload["spatial_residual"].shape) == [16, 392]
    assert list(payload["residual_abs"].shape) == [16, 392]
    assert 0.0 <= payload["residual_energy_ratio"] <= 1.0
    assert 0.0 <= payload["temporal_energy_ratio"] <= 1.0


def test_step38b_temporal_only_residual_energy_is_near_zero():
    temporal = torch.arange(16, dtype=torch.float32)[:, None].expand(16, 392)

    payload = decompose_proxy_importance(temporal)

    assert payload["residual_energy_ratio"] < 1.0e-8


def test_step38b_aggregate_keeps_metadata_safe():
    sample = {
        "sample_id": "sample",
        "trajectory_id": "traj",
        "shard_id": "shard1",
        "data_package_id": "fake",
        "importance": torch.rand(16, 392),
    }

    row = diagnose_step38b_sample(sample, [64, 128, 256, 512])
    summary = aggregate_label_diagnosis([row], "within_shard")

    assert row["metadata"]["future_tokens_loaded_for_diagnosis"] is False
    assert row["metadata"]["action_used_as_input"] is False
    assert row["metadata"]["language_used_as_input"] is False
    assert summary["num_rows"] == 1
