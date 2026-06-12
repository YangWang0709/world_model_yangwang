import json
from pathlib import Path

import pytest
import torch
import yaml

from data.bridgedata_v2_tfds_world_model_batch import load_bridge_tfds_world_model_samples


def _fake_config(tmp_path, missing_importance=False):
    token_path = tmp_path / "sample.pt"
    torch.save(
        {
            "sample_id": "sample",
            "trajectory_id": "traj",
            "context_tokens": torch.zeros((16, 392, 768), dtype=torch.float16),
            "current_tokens": torch.ones((4, 392, 768), dtype=torch.float16),
            "future_tokens": torch.full((4, 392, 768), 2.0, dtype=torch.float16),
            "metadata": {},
        },
        token_path,
    )
    token_manifest = tmp_path / "token_manifest.jsonl"
    token_manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "trajectory_id": "traj",
                "context_token_shape": [16, 392, 768],
                "current_token_shape": [4, 392, 768],
                "future_token_shape": [4, 392, 768],
                "token_artifact_path": str(token_path),
                "action_used_as_input": False,
                "language_used_as_input": False,
                "goal_used_as_input": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    importance_path = tmp_path / "importance.pt"
    torch.save(
        {
            "schema_version": "0.1.0",
            "stage": "bridgedata_v2_tfds_importance_step25",
            "sample_id": "sample",
            "trajectory_id": "traj",
            "method": "proxy_token_mse_dryrun",
            "context_importance_raw": torch.ones((16, 392)),
            "context_importance_norm": torch.ones((16, 392)),
            "temporal_importance": torch.ones(16),
            "spatial_importance": torch.ones(392),
            "stats": {},
            "metadata": {"action_used_as_input": False},
        },
        importance_path,
    )
    importance_manifest = tmp_path / "importance_manifest.jsonl"
    if missing_importance:
        importance_manifest.write_text("", encoding="utf-8")
    else:
        importance_manifest.write_text(
            json.dumps(
                {
                    "sample_id": "sample",
                    "trajectory_id": "traj",
                    "importance_artifact_path": str(importance_path),
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
            )
            + "\n",
            encoding="utf-8",
        )
    token_summary = tmp_path / "token_summary.json"
    token_summary.write_text(json.dumps({"token_extraction_performed": True}), encoding="utf-8")
    importance_summary = tmp_path / "importance_summary.json"
    importance_summary.write_text(json.dumps({"importance_generation_performed": True}), encoding="utf-8")
    return {
        "input": {
            "token_manifest_jsonl": str(token_manifest),
            "token_summary_json": str(token_summary),
            "token_smoke_dir": str(tmp_path),
            "importance_manifest_jsonl": str(importance_manifest),
            "importance_summary_json": str(importance_summary),
            "importance_smoke_dir": str(tmp_path),
        },
        "sample_limits": {"max_samples": 4},
    }


def test_fake_step24_and_step25_artifacts_align_by_sample_id(tmp_path):
    samples = load_bridge_tfds_world_model_samples(_fake_config(tmp_path))
    assert len(samples) == 1
    sample = samples[0]
    assert sample["sample_id"] == "sample"
    assert list(sample["context_tokens"].shape) == [16, 392, 768]
    assert list(sample["current_tokens"].shape) == [4, 392, 768]
    assert list(sample["future_tokens"].shape) == [4, 392, 768]
    assert list(sample["context_importance"].shape) == [16, 392]
    assert sample["metadata"]["current_tokens_kept_full"] is True
    assert sample["metadata"]["train_current_importance"] is False


def test_missing_importance_record_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing Step25 importance"):
        load_bridge_tfds_world_model_samples(_fake_config(tmp_path, missing_importance=True))
