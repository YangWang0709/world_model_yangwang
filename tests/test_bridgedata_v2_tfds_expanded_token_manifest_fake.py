from data.bridgedata_v2_tfds_expanded_token_manifest import (
    read_expanded_token_manifest,
    summarize_expanded_token_manifest,
    write_expanded_token_manifest,
)


def _token_record(sample_id="sample_0", split="train"):
    return {
        "sample_id": sample_id,
        "trajectory_id": "traj_0",
        "context_token_shape": [16, 392, 768],
        "current_token_shape": [4, 392, 768],
        "future_token_shape": [4, 392, 768],
        "token_artifact_path": f"/tmp/{sample_id}.pt",
        "image_field": "steps/observation/image_0",
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
        "trainval_split": split,
    }


def test_expanded_token_manifest_roundtrip_preserves_shapes_and_split(tmp_path):
    path = tmp_path / "token_manifest.jsonl"
    write_expanded_token_manifest([_token_record("a", "train"), _token_record("b", "val")], path)
    records = read_expanded_token_manifest(path)
    summary = summarize_expanded_token_manifest(records)
    assert summary["num_token_records"] == 2
    assert summary["context_token_shape_example"] == [16, 392, 768]
    assert summary["current_token_shape_example"] == [4, 392, 768]
    assert summary["future_token_shape_example"] == [4, 392, 768]
    assert summary["split_counts"] == {"train": 1, "val": 1}

