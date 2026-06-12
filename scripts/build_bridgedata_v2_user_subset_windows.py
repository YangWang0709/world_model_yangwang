"""Build Step22 user-subset windows when a valid user subset is present."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_manifest_schema import load_bridgedata_manifest_jsonl, write_bridgedata_manifest_jsonl
from data.bridgedata_v2_user_subset_ingestion import (
    inspect_user_subset_directory,
    write_user_subset_inspection_outputs,
)
from data.bridgedata_v2_user_subset_manifest_tools import (
    build_or_normalize_user_subset_manifest,
    write_manifest_summary,
)
from data.bridgedata_v2_user_subset_validator import validate_user_subset_manifest, write_validation_outputs
from data.bridgedata_v2_window_builder import (
    build_bridgedata_windows_from_manifest,
    summarize_manifest_window_build,
    write_window_manifest_jsonl,
)
from data.long_context_window_spec import window_spec_from_dict

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_user_subset_ingestion_step22.yaml"


def build_user_subset_windows_from_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    output = config["output"]
    spec = window_spec_from_dict(config["window"])

    inspection = inspect_user_subset_directory(
        dataset["user_subset_root"],
        manifest_path=dataset["manifest_path"],
        max_trajectories=int(dataset["max_trajectories_for_validation"]),
    )
    write_user_subset_inspection_outputs(inspection, output["subset_inspection_json"], output["subset_inspection_md"])

    manifest_summary = build_or_normalize_user_subset_manifest(
        dataset["user_subset_root"],
        output["generated_manifest_jsonl"],
        manifest_path=dataset["manifest_path"],
        min_frames=int(dataset["min_frames_per_trajectory"]),
        max_trajectories=int(dataset["max_trajectories_for_validation"]),
    )
    write_manifest_summary(manifest_summary, output["manifest_summary_json"])

    records = []
    manifest_path = Path(output["generated_manifest_jsonl"])
    if manifest_summary["generated_manifest_exists"] and manifest_path.exists():
        records = load_bridgedata_manifest_jsonl(manifest_path)

    validation = validate_user_subset_manifest(
        records,
        dataset["user_subset_root"],
        user_subset_exists=bool(inspection["user_subset_exists"]),
        min_frames=int(dataset["min_frames_per_trajectory"]),
        max_images_to_check_exists=int(dataset["max_images_to_check_exists"]),
        max_images_to_open_optional=int(dataset["max_images_to_open_optional"]),
        open_images=False,
    )
    write_validation_outputs(validation, output["validation_summary_json"], output["validation_summary_md"])

    window_manifest = Path(output["user_window_manifest_jsonl"])
    if not validation["real_format_validated"] and window_manifest.exists():
        window_manifest.unlink()
    windows: list[dict[str, Any]] = []
    if validation["real_format_validated"]:
        valid_ids = set(validation["valid_trajectory_ids"])
        valid_records = [record for record in records if record["trajectory_id"] in valid_ids]
        normalized_valid_manifest = Path(output["generated_manifest_jsonl"])
        write_bridgedata_manifest_jsonl(normalized_valid_manifest, valid_records)
        windows = build_bridgedata_windows_from_manifest(valid_records, spec)
        write_window_manifest_jsonl(window_manifest, windows)
    window_summary = summarize_manifest_window_build(records, windows, spec) if records else _empty_window_summary(spec)
    summary = {
        "stage": "bridgedata_v2_user_subset_ingestion_step22",
        "user_subset_exists": bool(inspection["user_subset_exists"]),
        "pending_user_data": bool(inspection["pending_user_data"] or manifest_summary["pending_user_data"]),
        "real_format_validated": bool(validation["real_format_validated"]),
        "manifest_exists": bool(inspection["manifest_exists"]),
        "generated_manifest_exists": bool(manifest_summary["generated_manifest_exists"]),
        "user_window_manifest_exists": bool(window_manifest.exists()),
        "num_manifest_records": int(manifest_summary["num_manifest_records"]),
        "num_valid_trajectories": int(validation["num_valid_trajectories"]),
        "num_skipped_trajectories": int(validation["num_skipped_trajectories"]),
        "num_windows": len(windows),
        "context_len": spec.context_len,
        "current_len": spec.current_len,
        "future_len": spec.future_len,
        "frame_count_min": validation["frame_count_min"],
        "frame_count_mean": validation["frame_count_mean"],
        "frame_count_max": validation["frame_count_max"],
        "has_images_likely": validation["has_images_likely"],
        "has_actions_likely": validation["has_actions_likely"],
        "has_language_likely": validation["has_language_likely"],
        "has_goal_image_likely": validation["has_goal_image_likely"],
        "download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "use_action_as_input": False,
        "use_language_as_input": False,
        "use_goal_image_as_input": False,
        "safety_gate_pass": bool(validation["safety_gate_pass"]),
        "window_manifest_path": str(window_manifest),
        "window_build_summary": window_summary,
        "inspection": inspection,
        "manifest_summary": manifest_summary,
        "validation": validation,
    }
    summary_path = Path(output["window_builder_summary_json"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _empty_window_summary(spec) -> dict[str, Any]:
    return {
        "num_trajectories": 0,
        "num_valid_trajectories": 0,
        "num_skipped_trajectories": 0,
        "skipped_trajectories": [],
        "num_windows": 0,
        "context_len": spec.context_len,
        "current_len": spec.current_len,
        "future_len": spec.future_len,
        "required_total_frames": spec.required_total_frames,
        "window_summary": {
            "num_windows": 0,
            "num_trajectories": 0,
            "context_len": None,
            "current_len": None,
            "future_len": None,
            "has_frame_path_refs": False,
            "action_used_as_input": False,
            "language_used_as_input": False,
            "goal_image_used_as_input": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(build_user_subset_windows_from_config(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
