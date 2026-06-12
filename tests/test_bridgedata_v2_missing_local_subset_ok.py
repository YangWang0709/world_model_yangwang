from pathlib import Path

import yaml

from scripts.build_bridgedata_v2_context_windows import build_from_config
from scripts.inspect_bridgedata_v2_tiny_subset import inspect_from_config


def _write_tmp_config(tmp_path):
    config = yaml.safe_load(Path("configs/bridgedata_v2_tiny_window_builder_step20.yaml").read_text(encoding="utf-8"))
    missing = tmp_path / "missing_subset"
    run_dir = tmp_path / "run"
    config["dataset"]["local_subset_candidates"] = [str(missing)]
    config["dataset"]["manifest_candidates"] = [str(missing / "manifest.jsonl")]
    config["output_records"]["output_root"] = str(run_dir)
    config["output_records"]["window_manifest_jsonl"] = str(run_dir / "bridgedata_v2_window_manifest.jsonl")
    config["output_records"]["trajectory_summary_json"] = str(run_dir / "trajectory_summary.json")
    config["output_records"]["builder_summary_json"] = str(run_dir / "bridgedata_v2_window_builder_summary.json")
    config["output_records"]["builder_summary_md"] = str(run_dir / "bridgedata_v2_window_builder_summary.md")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_missing_local_subset_uses_fake_manifest_and_still_builds_windows(tmp_path):
    config_path = _write_tmp_config(tmp_path)
    inspection = inspect_from_config(config_path)
    assert inspection["local_subset_exists"] is False
    assert inspection["fake_manifest_available"] is True
    summary = build_from_config(config_path)
    assert summary["local_subset_exists"] is False
    assert summary["used_fake_manifest"] is True
    assert summary["num_windows"] > 0
    assert Path(summary["window_manifest_jsonl"]).exists()
