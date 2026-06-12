import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_tfds_mini_shard import evaluate_tfds_mini_shard


def _config_and_outputs(tmp_path: Path, *, validated: bool) -> Path:
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_mini_shard_step23.yaml").read_text(encoding="utf-8"))
    run = tmp_path / "run"
    run.mkdir(parents=True)
    output_keys = {
        "inventory_json": "inventory.json",
        "download_summary_json": "download.json",
        "rlds_schema_json": "schema.json",
        "tfds_manifest_summary_json": "manifest_summary.json",
        "tfds_window_summary_json": "window_summary.json",
        "tfds_manifest_jsonl": "manifest.jsonl",
        "tfds_window_manifest_jsonl": "windows.jsonl",
        "eval_json": "eval.json",
        "eval_md": "eval.md",
    }
    for key, name in output_keys.items():
        config["output"][key] = str(run / name)
    Path(config["output"]["inventory_json"]).write_text(
        json.dumps({"safe_stop": False, "num_train_shards_found": 10, "selected_shards": [], "selected_total_bytes": 0}),
        encoding="utf-8",
    )
    Path(config["output"]["download_summary_json"]).write_text(
        json.dumps(
            {
                "safe_stop": not validated,
                "reason": None if validated else "download safe-stop",
                "download_performed": validated,
                "download_bytes": 100,
                "num_shards_downloaded": 1 if validated else 0,
                "raw_zip_downloaded": False,
                "full_tfds_downloaded": False,
                "droid_downloaded": False,
            }
        ),
        encoding="utf-8",
    )
    Path(config["output"]["rlds_schema_json"]).write_text(
        json.dumps(
            {
                "safe_stop": not validated,
                "reason": None if validated else "schema safe-stop",
                "tfds_env_ok": True,
                "dataset_root_exists": True,
                "num_episodes_scanned": 2 if validated else 0,
                "can_build_16_4_4_windows": validated,
                "candidate_fields": {"image_fields": ["steps/observation/image_0"], "action_fields": ["steps/action"]},
            }
        ),
        encoding="utf-8",
    )
    Path(config["output"]["tfds_manifest_summary_json"]).write_text(
        json.dumps({"num_manifest_records": 1 if validated else 0, "safe_stop": not validated}),
        encoding="utf-8",
    )
    Path(config["output"]["tfds_window_summary_json"]).write_text(
        json.dumps(
            {
                "real_tfds_validated": validated,
                "safe_stop": not validated,
                "reason": None if validated else "no windows",
                "num_valid_trajectories": 1 if validated else 0,
                "num_windows": 7 if validated else 0,
                "training_performed": False,
                "token_extraction_performed": False,
                "importance_generation_performed": False,
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_validated_report_recommends_step24_token_dry_run(tmp_path):
    docs_dir = tmp_path / "docs"
    summary = evaluate_tfds_mini_shard(_config_and_outputs(tmp_path, validated=True), docs_dir=docs_dir)
    report = (docs_dir / "BRIDGEDATA_V2_TFDS_MINI_SHARD_REPORT.md").read_text(encoding="utf-8")
    assert summary["real_tfds_validated"] is True
    assert summary["recommended_step24"]["name"] == "BridgeData V2 TFDS mini-shard token extraction dry-run"
    assert "TFDS/RLDS mini-shard smoke succeeded." in report


def test_safe_stop_report_does_not_claim_validation(tmp_path):
    docs_dir = tmp_path / "docs"
    summary = evaluate_tfds_mini_shard(_config_and_outputs(tmp_path, validated=False), docs_dir=docs_dir)
    report = (docs_dir / "BRIDGEDATA_V2_TFDS_MINI_SHARD_REPORT.md").read_text(encoding="utf-8")
    assert summary["real_tfds_validated"] is False
    assert summary["safe_stop"] is True
    assert "TFDS/RLDS mini-shard smoke did not validate real episodes." in report
    assert "TFDS/RLDS mini-shard smoke succeeded." not in report
