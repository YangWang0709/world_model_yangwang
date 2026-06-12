"""Build LongContextSample windows from the Step23 TFDS mini manifest."""

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

from data.bridgedata_v2_manifest_schema import load_bridgedata_manifest_jsonl
from data.bridgedata_v2_window_builder import (
    build_bridgedata_windows_from_manifest,
    summarize_manifest_window_build,
    write_window_manifest_jsonl,
)
from data.long_context_window_spec import window_spec_from_dict

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_mini_shard_step23.yaml"


def build_windows_from_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(config["output"]["tfds_manifest_jsonl"])
    window_path = Path(config["output"]["tfds_window_manifest_jsonl"])
    manifest_summary_path = Path(config["output"]["tfds_manifest_summary_json"])
    manifest_summary = json.loads(manifest_summary_path.read_text(encoding="utf-8")) if manifest_summary_path.exists() else {}
    spec = window_spec_from_dict(config["window"])
    records = load_bridgedata_manifest_jsonl(manifest_path) if manifest_path.exists() else []
    windows: list[dict[str, Any]] = []
    if records and not manifest_summary.get("safe_stop"):
        windows = build_bridgedata_windows_from_manifest(records, spec, max_windows=int(config["sample_limits"]["max_windows"]))
        write_window_manifest_jsonl(window_path, windows)
    elif window_path.exists():
        window_path.unlink()
    window_summary = summarize_manifest_window_build(records, windows, spec) if records else _empty_window_summary(spec)
    real_tfds_validated = bool(records and windows and not manifest_summary.get("safe_stop"))
    summary = {
        "stage": "bridgedata_v2_tfds_mini_window_builder",
        "real_tfds_validated": real_tfds_validated,
        "safe_stop": not real_tfds_validated,
        "reason": None if real_tfds_validated else manifest_summary.get("reason") or "No TFDS mini windows were generated.",
        "num_manifest_records": len(records),
        "num_valid_trajectories": window_summary["num_valid_trajectories"],
        "num_skipped_trajectories": window_summary["num_skipped_trajectories"],
        "num_windows": len(windows),
        "context_len": spec.context_len,
        "current_len": spec.current_len,
        "future_len": spec.future_len,
        "tfds_mini_manifest_path": str(manifest_path),
        "tfds_mini_window_manifest_path": str(window_path),
        "tfds_mini_window_manifest_exists": window_path.exists(),
        "raw_zip_downloaded": False,
        "full_tfds_downloaded": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "safety_gate_pass": True,
        "window_summary": window_summary,
    }
    output = Path(config["output"]["tfds_window_summary_json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
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
    print(json.dumps(build_windows_from_config(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
