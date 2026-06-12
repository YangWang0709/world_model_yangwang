import torch

from data.bridgedata_v2_tfds_importance_manifest import (
    read_importance_manifest_jsonl,
    summarize_importance_manifest,
    validate_importance_artifact,
    write_importance_manifest_jsonl,
)


def test_fake_importance_manifest_roundtrip_and_artifact_validation(tmp_path):
    artifact_path = tmp_path / "importance.pt"
    torch.save(
        {
            "schema_version": "0.1.0",
            "stage": "bridgedata_v2_tfds_importance_step25",
            "sample_id": "sample",
            "trajectory_id": "traj",
            "method": "proxy_token_mse_dryrun",
            "context_importance_raw": torch.zeros((16, 392)),
            "context_importance_norm": torch.zeros((16, 392)),
            "temporal_importance": torch.zeros(16),
            "spatial_importance": torch.zeros(392),
            "stats": {},
            "metadata": {
                "action_used_as_input": False,
                "language_used_as_input": False,
                "goal_used_as_input": False,
            },
        },
        artifact_path,
    )
    record = {
        "sample_id": "sample",
        "trajectory_id": "traj",
        "importance_artifact_path": str(artifact_path),
        "method": "proxy_token_mse_dryrun",
        "context_importance_shape": [16, 392],
        "temporal_importance_shape": [16],
        "spatial_importance_shape": [392],
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
    }
    manifest_path = tmp_path / "manifest.jsonl"
    write_importance_manifest_jsonl([record], manifest_path)
    assert read_importance_manifest_jsonl(manifest_path) == [record]
    assert validate_importance_artifact(artifact_path)["method"] == "proxy_token_mse_dryrun"
    summary = summarize_importance_manifest([record])
    assert summary["num_samples"] == 1
    assert summary["context_importance_shape_example"] == [16, 392]
