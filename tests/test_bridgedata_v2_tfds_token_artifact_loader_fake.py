import json

import pytest
import torch

from data.bridgedata_v2_tfds_token_artifact_loader import (
    EXPECTED_TOKEN_SHAPES,
    iter_step24_token_samples,
    load_step24_token_artifact,
    load_step24_token_manifest,
)


def _write_fake_token_artifact(tmp_path, context_shape=None):
    context_shape = context_shape or EXPECTED_TOKEN_SHAPES["context"]
    artifact_path = tmp_path / "sample.pt"
    torch.save(
        {
            "schema_version": "0.1.0",
            "stage": "bridgedata_v2_tfds_token_extraction_step24",
            "sample_id": "sample",
            "trajectory_id": "traj",
            "context_tokens": torch.zeros(context_shape, dtype=torch.float16),
            "current_tokens": torch.zeros(EXPECTED_TOKEN_SHAPES["current"], dtype=torch.float16),
            "future_tokens": torch.ones(EXPECTED_TOKEN_SHAPES["future"], dtype=torch.float16),
            "metadata": {"language": "metadata only"},
        },
        artifact_path,
    )
    record = {
        "sample_id": "sample",
        "trajectory_id": "traj",
        "context_token_shape": context_shape,
        "current_token_shape": EXPECTED_TOKEN_SHAPES["current"],
        "future_token_shape": EXPECTED_TOKEN_SHAPES["future"],
        "token_artifact_path": str(artifact_path),
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
    }
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return record, manifest_path


def test_fake_step24_token_artifact_loads_as_cpu_float32(tmp_path):
    record, manifest_path = _write_fake_token_artifact(tmp_path)
    records = load_step24_token_manifest(manifest_path)
    assert records == [record]
    sample = load_step24_token_artifact(record)
    assert list(sample["context_tokens"].shape) == EXPECTED_TOKEN_SHAPES["context"]
    assert list(sample["current_tokens"].shape) == EXPECTED_TOKEN_SHAPES["current"]
    assert list(sample["future_tokens"].shape) == EXPECTED_TOKEN_SHAPES["future"]
    assert sample["context_tokens"].dtype == torch.float32
    assert sample["context_tokens"].device.type == "cpu"
    assert sample["action_used_as_input"] is False
    assert sample["language_used_as_input"] is False
    assert sample["goal_used_as_input"] is False
    assert len(list(iter_step24_token_samples(manifest_path, max_samples=1))) == 1


def test_shape_mismatch_raises(tmp_path):
    record, _ = _write_fake_token_artifact(tmp_path, context_shape=[15, 392, 768])
    with pytest.raises(ValueError, match="context_token_shape"):
        load_step24_token_artifact(record)


def test_input_flags_are_rejected(tmp_path):
    record, _ = _write_fake_token_artifact(tmp_path)
    record["action_used_as_input"] = True
    with pytest.raises(ValueError, match="action_used_as_input"):
        load_step24_token_artifact(record)
