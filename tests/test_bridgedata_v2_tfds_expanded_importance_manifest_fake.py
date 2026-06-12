from data.bridgedata_v2_tfds_expanded_importance_manifest import (
    read_expanded_importance_manifest,
    summarize_expanded_importance_manifest,
    write_expanded_importance_manifest,
)


def _importance_record(sample_id="sample_0", split="train"):
    return {
        "sample_id": sample_id,
        "trajectory_id": "traj_0",
        "importance_artifact_path": f"/tmp/{sample_id}_importance.pt",
        "method": "proxy_token_mse_dryrun",
        "context_importance_shape": [16, 392],
        "temporal_importance_shape": [16],
        "spatial_importance_shape": [392],
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "current_importance_generated": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
        "trainval_split": split,
    }


def test_expanded_importance_manifest_roundtrip_preserves_context_only_importance(tmp_path):
    path = tmp_path / "importance_manifest.jsonl"
    write_expanded_importance_manifest([_importance_record("a", "train"), _importance_record("b", "val")], path)
    records = read_expanded_importance_manifest(path)
    summary = summarize_expanded_importance_manifest(records)
    assert summary["num_importance_records"] == 2
    assert summary["context_importance_shape_example"] == [16, 392]
    assert summary["current_importance_generated"] is False
    assert summary["split_counts"] == {"train": 1, "val": 1}

