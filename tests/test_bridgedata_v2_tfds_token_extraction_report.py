import json
from pathlib import Path

import yaml

import eval.eval_bridgedata_v2_tfds_token_extraction as report_module


def _config(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    config = {
        "output": {
            "clip_export_summary_json": str(run / "clip_export_summary.json"),
            "token_summary_json": str(run / "tfds_token_extraction_summary.json"),
            "token_manifest_jsonl": str(run / "tfds_token_smoke_manifest.jsonl"),
            "eval_json": str(run / "tfds_token_extraction_eval.json"),
            "eval_md": str(run / "tfds_token_extraction_eval.md"),
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_recommends_step25_on_success(tmp_path: Path, monkeypatch):
    config_path = _config(tmp_path)
    run = tmp_path / "run"
    monkeypatch.setattr(report_module, "DOC_REPORT", tmp_path / "doc.md")
    _write_json(
        run / "clip_export_summary.json",
        {"clip_export_performed": True, "safe_stop": False, "num_windows_exported": 1, "safety_gate_pass": True},
    )
    _write_json(
        run / "tfds_token_extraction_summary.json",
        {
            "token_extraction_performed": True,
            "safe_stop": False,
            "model_missing": False,
            "image_field": "steps/observation/image_0",
            "model_loaded_local_only": True,
            "model_download_performed": False,
            "training_performed": False,
            "importance_generation_performed": False,
            "large_token_shards_generated": False,
            "data_token_shards_written": False,
            "safety_gate_pass": True,
        },
    )
    (run / "tfds_token_smoke_manifest.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample_0",
                "trajectory_id": "tfds_episode_000000",
                "context_token_shape": [16, 196, 768],
                "current_token_shape": [4, 196, 768],
                "future_token_shape": [4, 196, 768],
                "token_artifact_path": str(run / "sample_0.pt"),
                "image_field": "steps/observation/image_0",
                "action_used_as_input": False,
                "language_used_as_input": False,
                "goal_used_as_input": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = report_module.evaluate_step24_token_extraction(config_path)
    assert result["pass"] is True
    assert result["recommended_step25"]["name"] == "BridgeData V2 TFDS mini-shard predictive importance dry-run"
    assert result["safety_gate_pass"] is True


def test_report_does_not_call_model_missing_safe_stop_success(tmp_path: Path, monkeypatch):
    config_path = _config(tmp_path)
    run = tmp_path / "run"
    monkeypatch.setattr(report_module, "DOC_REPORT", tmp_path / "doc.md")
    _write_json(
        run / "clip_export_summary.json",
        {"clip_export_performed": True, "safe_stop": False, "num_windows_exported": 1, "safety_gate_pass": True},
    )
    _write_json(
        run / "tfds_token_extraction_summary.json",
        {
            "token_extraction_performed": False,
            "safe_stop": True,
            "model_missing": True,
            "model_download_performed": False,
            "training_performed": False,
            "importance_generation_performed": False,
            "large_token_shards_generated": False,
            "data_token_shards_written": False,
            "safety_gate_pass": True,
        },
    )
    result = report_module.evaluate_step24_token_extraction(config_path)
    assert result["pass"] is False
    assert result["safe_stop"] is True
    assert result["model_missing"] is True
    assert result["recommended_step25"]["name"] == "Fix local VideoMAE cache / restore local model before token extraction"
    assert result["safety_gate_pass"] is True
