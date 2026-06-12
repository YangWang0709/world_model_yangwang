import json
from pathlib import Path

import torch

from data.bridgedata_v2_tfds_true_temporal_schema import (
    TOKENIZATION_MODE,
    load_true_temporal_token_artifact,
    read_true_temporal_manifest,
    summarize_tokens,
    target_summary_for_variant,
    validate_true_temporal_sample,
    write_true_temporal_manifest,
)


def test_true_temporal_schema_is_shape_flexible_and_delta_targets_are_readable(tmp_path: Path):
    artifact_path = tmp_path / "sample.pt"
    context = torch.randn(7, 768)
    current = torch.randn(5, 768)
    future = torch.randn(9, 768)
    torch.save(
        {
            "sample_id": "s0",
            "trajectory_id": "traj0",
            "horizon_gap": 0,
            "tokenization_mode": TOKENIZATION_MODE,
            "context_tokens": context,
            "current_tokens": current,
            "future_tokens": future,
            "raw_token_shapes": {"context": [1, 7, 768], "current": [1, 5, 768], "future": [1, 9, 768]},
            "metadata": {"action_used_as_input": False, "language_used_as_input": False, "goal_used_as_input": False},
        },
        artifact_path,
    )
    manifest = [
        {
            "sample_id": "s0",
            "trajectory_id": "traj0",
            "horizon_gap": 0,
            "tokenization_mode": TOKENIZATION_MODE,
            "context_token_shape": [7, 768],
            "current_token_shape": [5, 768],
            "future_token_shape": [9, 768],
            "raw_token_shapes": {"context": [1, 7, 768], "current": [1, 5, 768], "future": [1, 9, 768]},
            "token_artifact_path": str(artifact_path),
            "image_field": "steps/observation/image_0",
            "action_used_as_input": False,
            "language_used_as_input": False,
            "goal_used_as_input": False,
        }
    ]
    path = write_true_temporal_manifest(manifest, tmp_path / "manifest.jsonl")
    assert read_true_temporal_manifest(path)[0]["context_token_shape"] == [7, 768]
    sample = load_true_temporal_token_artifact(manifest[0])
    assert validate_true_temporal_sample(sample) is True
    assert list(summarize_tokens(sample["context_tokens"]).shape) == [768]
    assert list(target_summary_for_variant(sample, "future_delta_last_minus_current").shape) == [768]
    assert list(sample["context_tokens"].shape) != [16, 392, 768]


def test_true_temporal_manifest_rejects_action_as_input(tmp_path: Path):
    bad = {
        "sample_id": "s0",
        "trajectory_id": "traj0",
        "horizon_gap": 0,
        "tokenization_mode": TOKENIZATION_MODE,
        "context_token_shape": [7, 768],
        "current_token_shape": [5, 768],
        "future_token_shape": [9, 768],
        "raw_token_shapes": {"context": [1, 7, 768], "current": [1, 5, 768], "future": [1, 9, 768]},
        "token_artifact_path": str(tmp_path / "missing.pt"),
        "action_used_as_input": True,
        "language_used_as_input": False,
        "goal_used_as_input": False,
    }
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    try:
        read_true_temporal_manifest(path)
    except ValueError as exc:
        assert "action_used_as_input" in str(exc)
    else:
        raise AssertionError("manifest with action input should fail")
