import json
from pathlib import Path

import torch
import yaml

from scripts.run_bridgedata_v2_tfds_world_model_smoke import run_bridge_tfds_world_model_smoke


def _write_fake_inputs(tmp_path):
    token_artifact = tmp_path / "token.pt"
    torch.save(
        {
            "sample_id": "sample",
            "trajectory_id": "traj",
            "context_tokens": torch.zeros((16, 392, 768), dtype=torch.float16),
            "current_tokens": torch.ones((4, 392, 768), dtype=torch.float16),
            "future_tokens": torch.ones((4, 392, 768), dtype=torch.float16),
            "metadata": {},
        },
        token_artifact,
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
                "token_artifact_path": str(token_artifact),
                "action_used_as_input": False,
                "language_used_as_input": False,
                "goal_used_as_input": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    importance_artifact = tmp_path / "importance.pt"
    torch.save(
        {
            "method": "proxy_token_mse_dryrun",
            "context_importance_raw": torch.ones((16, 392)),
            "context_importance_norm": torch.linspace(0.0, 1.0, 16 * 392).reshape(16, 392),
            "temporal_importance": torch.ones(16),
            "spatial_importance": torch.ones(392),
            "metadata": {"action_used_as_input": False},
        },
        importance_artifact,
    )
    importance_manifest = tmp_path / "importance_manifest.jsonl"
    importance_manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "trajectory_id": "traj",
                "importance_artifact_path": str(importance_artifact),
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
    return token_manifest, token_summary, importance_manifest, importance_summary


def _config(tmp_path, token_manifest, token_summary, importance_manifest, importance_summary):
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_world_model_smoke_step26.yaml").read_text(encoding="utf-8"))
    config["input"]["token_manifest_jsonl"] = str(token_manifest)
    config["input"]["token_summary_json"] = str(token_summary)
    config["input"]["token_smoke_dir"] = str(tmp_path)
    config["input"]["importance_manifest_jsonl"] = str(importance_manifest)
    config["input"]["importance_summary_json"] = str(importance_summary)
    config["input"]["importance_smoke_dir"] = str(tmp_path)
    for key in [
        "batch_summary_json",
        "policy_metrics_json",
        "forward_loss_json",
        "smoke_summary_json",
        "smoke_summary_md",
        "eval_json",
        "eval_md",
    ]:
        suffix = ".md" if key.endswith("_md") else ".json"
        config["output"][key] = str(tmp_path / f"{key}{suffix}")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_run_script_reads_existing_artifacts_without_training_download_or_generation(tmp_path):
    inputs = _write_fake_inputs(tmp_path)
    config_path = _config(tmp_path, *inputs)
    result = run_bridge_tfds_world_model_smoke(config_path)
    assert result["world_model_smoke_performed"] is True
    assert result["download_performed"] is False
    assert result["token_extraction_performed"] is False
    assert result["importance_generation_performed"] is False
    assert result["training_performed"] is False
    assert result["optimizer_step_performed"] is False
    assert result["data_token_shards_written"] is False
    assert result["data_importance_shards_written"] is False
    assert result["all_losses_finite"] is True
    output_text = json.dumps(result)
    assert "data/token_shards" not in output_text
    assert "data/importance_shards" not in output_text
