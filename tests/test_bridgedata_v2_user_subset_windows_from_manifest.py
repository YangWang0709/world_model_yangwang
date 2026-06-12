import json
from pathlib import Path

import yaml

from data.bridgedata_v2_window_builder import load_window_manifest_jsonl
from data.long_context_dataset_schema import validate_long_context_sample
from scripts.build_bridgedata_v2_user_subset_windows import build_user_subset_windows_from_config


def _write_config(tmp_path: Path, subset: Path) -> Path:
    config = yaml.safe_load(Path("configs/bridgedata_v2_user_subset_ingestion_step22.yaml").read_text(encoding="utf-8"))
    run = tmp_path / "run"
    config["dataset"]["user_subset_root"] = str(subset)
    config["dataset"]["manifest_path"] = str(subset / "manifest.jsonl")
    config["template"]["output_dir"] = str(run / "template")
    config["output"]["subset_inspection_json"] = str(run / "inspection.json")
    config["output"]["subset_inspection_md"] = str(run / "inspection.md")
    config["output"]["generated_manifest_jsonl"] = str(run / "manifest.jsonl")
    config["output"]["manifest_summary_json"] = str(run / "manifest_summary.json")
    config["output"]["user_window_manifest_jsonl"] = str(run / "windows.jsonl")
    config["output"]["window_builder_summary_json"] = str(run / "window_summary.json")
    config["output"]["validation_summary_json"] = str(run / "validation.json")
    config["output"]["validation_summary_md"] = str(run / "validation.md")
    config["output"]["eval_json"] = str(run / "eval.json")
    config["output"]["eval_md"] = str(run / "eval.md")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_fake_manifest_generates_user_subset_windows(tmp_path):
    subset = tmp_path / "subset"
    images = subset / "traj_000001" / "images"
    images.mkdir(parents=True)
    frame_paths = []
    for idx in range(30):
        path = images / f"frame_{idx:06d}.jpg"
        path.write_text("placeholder", encoding="utf-8")
        frame_paths.append(f"traj_000001/images/frame_{idx:06d}.jpg")
    (subset / "manifest.jsonl").write_text(
        json.dumps(
            {
                "trajectory_id": "traj_000001",
                "num_frames": 30,
                "frame_paths": frame_paths,
                "actions_path": "traj_000001/actions.npy",
                "language_instruction": "move object",
                "goal_image_path": "traj_000001/goal.jpg",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, subset)
    summary = build_user_subset_windows_from_config(config_path)
    assert summary["real_format_validated"] is True
    assert summary["user_window_manifest_exists"] is True
    assert summary["context_len"] == 16
    assert summary["current_len"] == 4
    assert summary["future_len"] == 4
    assert summary["num_windows"] == 7
    windows = load_window_manifest_jsonl(tmp_path / "run" / "windows.jsonl")
    assert len(windows) == 7
    for window in windows:
        assert validate_long_context_sample(window)
        assert window["metadata"]["use_action_as_input"] is False
        assert window["metadata"]["use_language_as_input"] is False
        assert window["metadata"]["use_goal_image_as_input"] is False
