import json
from pathlib import Path

import torch
import yaml

from scripts.generate_bridgedata_v2_tfds_importance_dryrun import (
    generate_bridgedata_v2_tfds_importance_from_config,
)


def _write_fake_step24(tmp_path):
    token_dir = tmp_path / "token_smoke"
    token_dir.mkdir()
    artifact_path = token_dir / "sample.pt"
    torch.save(
        {
            "schema_version": "0.1.0",
            "stage": "bridgedata_v2_tfds_token_extraction_step24",
            "sample_id": "sample",
            "trajectory_id": "traj",
            "context_tokens": torch.zeros((16, 392, 768), dtype=torch.float16),
            "current_tokens": torch.zeros((4, 392, 768), dtype=torch.float16),
            "future_tokens": torch.ones((4, 392, 768), dtype=torch.float16),
            "metadata": {},
        },
        artifact_path,
    )
    manifest = tmp_path / "tfds_token_smoke_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "trajectory_id": "traj",
                "context_token_shape": [16, 392, 768],
                "current_token_shape": [4, 392, 768],
                "future_token_shape": [4, 392, 768],
                "token_artifact_path": str(artifact_path),
                "action_used_as_input": False,
                "language_used_as_input": False,
                "goal_used_as_input": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = tmp_path / "tfds_token_extraction_summary.json"
    summary.write_text(json.dumps({"token_extraction_performed": True}), encoding="utf-8")
    return manifest, summary, token_dir


def _write_config(tmp_path, manifest, summary, token_dir):
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_importance_step25.yaml").read_text(encoding="utf-8"))
    config["input"]["token_manifest_jsonl"] = str(manifest)
    config["input"]["token_summary_json"] = str(summary)
    config["input"]["token_smoke_dir"] = str(token_dir)
    config["output"]["run_root"] = str(tmp_path)
    config["output"]["importance_dir"] = str(tmp_path / "importance_smoke")
    config["output"]["importance_manifest_jsonl"] = str(tmp_path / "tfds_importance_smoke_manifest.jsonl")
    config["output"]["importance_summary_json"] = str(tmp_path / "tfds_importance_summary.json")
    config["output"]["importance_summary_md"] = str(tmp_path / "tfds_importance_summary.md")
    config["output"]["eval_json"] = str(tmp_path / "tfds_importance_eval.json")
    config["output"]["eval_md"] = str(tmp_path / "tfds_importance_eval.md")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_generation_uses_existing_tokens_without_download_training_or_token_extraction(tmp_path):
    manifest, summary, token_dir = _write_fake_step24(tmp_path)
    config_path = _write_config(tmp_path, manifest, summary, token_dir)
    result = generate_bridgedata_v2_tfds_importance_from_config(config_path)
    assert result["importance_generation_performed"] is True
    assert result["download_performed"] is False
    assert result["model_download_performed"] is False
    assert result["training_performed"] is False
    assert result["teacher_training_performed"] is False
    assert result["selector_training_performed"] is False
    assert result["world_model_training_performed"] is False
    assert result["token_extraction_performed"] is False
    assert result["data_token_shards_written"] is False
    assert result["data_importance_shards_written"] is False
    assert result["context_importance_shape_example"] == [16, 392]
    output_text = json.dumps(result)
    assert "data/token_shards" not in output_text
    assert "data/importance_shards" not in output_text
