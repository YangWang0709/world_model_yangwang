from pathlib import Path

from data.bridgedata_v2_tfds_token_manifest import (
    read_token_manifest_jsonl,
    summarize_token_manifest,
    write_token_manifest_jsonl,
)


def test_token_manifest_round_trip_and_summary(tmp_path: Path):
    records = [
        {
            "sample_id": "sample_0",
            "trajectory_id": "tfds_episode_000000",
            "context_token_shape": [16, 196, 768],
            "current_token_shape": [4, 196, 768],
            "future_token_shape": [4, 196, 768],
            "token_artifact_path": str(tmp_path / "sample_0.pt"),
            "image_field": "steps/observation/image_0",
            "action_used_as_input": False,
            "language_used_as_input": False,
            "goal_used_as_input": False,
        }
    ]
    path = tmp_path / "manifest.jsonl"
    write_token_manifest_jsonl(records, path)
    loaded = read_token_manifest_jsonl(path)
    assert loaded == records
    summary = summarize_token_manifest(loaded)
    assert summary["num_windows_tokenized"] == 1
    assert summary["context_token_shape_example"] == [16, 196, 768]
    assert loaded[0]["action_used_as_input"] is False
    assert loaded[0]["language_used_as_input"] is False
