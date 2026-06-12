import json
from pathlib import Path

import yaml

from data.bridgedata_v2_real_format_validator import validate_bridgedata_v2_real_tiny_subset


def _base_config(tmp_path, subset):
    config = yaml.safe_load(Path("configs/bridgedata_v2_real_tiny_validation_step21.yaml").read_text(encoding="utf-8"))
    run = tmp_path / "run"
    config["dataset"]["subset_root"] = str(subset)
    config["dataset"]["download_root"] = str(tmp_path / "downloads")
    config["dataset"]["extract_root"] = str(tmp_path / "extract")
    config["output"]["real_manifest_jsonl"] = str(run / "real_tiny_manifest.jsonl")
    config["output"]["real_window_manifest_jsonl"] = str(run / "real_tiny_window_manifest.jsonl")
    config["output"]["validation_summary_json"] = str(run / "real_tiny_validation_summary.json")
    config["output"]["validation_summary_md"] = str(run / "real_tiny_validation_summary.md")
    return config


def test_validator_accepts_existing_manifest_without_actions_language_goal(tmp_path):
    subset = tmp_path / "subset"
    subset.mkdir()
    frame_paths = [f"traj0/images/frame_{idx:06d}.jpg" for idx in range(30)]
    (subset / "manifest.jsonl").write_text(
        json.dumps({"trajectory_id": "traj0", "num_frames": 30, "frame_paths": frame_paths}) + "\n",
        encoding="utf-8",
    )
    config = _base_config(tmp_path, subset)
    acquisition = {"safe_stop": False, "reason": None, "extract_root": str(tmp_path / "extract")}
    summary = validate_bridgedata_v2_real_tiny_subset(config, acquisition)
    assert summary["real_format_validated"] is True
    assert summary["num_manifest_records"] == 1
    assert summary["num_windows"] == 7
    assert summary["use_action_as_input"] is False
    assert summary["use_language_as_input"] is False
    assert summary["use_goal_image_as_input"] is False


def test_validator_unknown_layout_safe_stops_with_warning(tmp_path):
    subset = tmp_path / "subset"
    subset.mkdir()
    (subset / "notes.txt").write_text("unknown layout", encoding="utf-8")
    config = _base_config(tmp_path, subset)
    acquisition = {"safe_stop": False, "reason": None, "extract_root": str(tmp_path / "extract")}
    summary = validate_bridgedata_v2_real_tiny_subset(config, acquisition)
    assert summary["real_format_validated"] is False
    assert summary["safe_stop"] is True
    assert summary["user_provided_subset_required"] is True
