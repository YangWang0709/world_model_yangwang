"""Build BridgeData V2 tiny-subset context/current/future window manifests."""

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

from data.bridgedata_v2_format_inspector import find_first_existing_manifest
from data.bridgedata_v2_manifest_schema import load_bridgedata_manifest_jsonl, make_fake_bridgedata_manifest_records
from data.bridgedata_v2_window_builder import (
    build_bridgedata_windows_from_manifest,
    summarize_manifest_window_build,
    write_window_manifest_jsonl,
)
from data.long_context_window_spec import LongContextWindowSpec, window_spec_from_dict
from scripts.inspect_bridgedata_v2_tiny_subset import inspect_from_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tiny_window_builder_step20.yaml"


def build_from_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(config["output_records"]["output_root"])
    run_dir.mkdir(parents=True, exist_ok=True)
    inspection = inspect_from_config(config_path)

    manifest = find_first_existing_manifest(config["dataset"].get("manifest_candidates", []))
    used_fake_manifest = False
    if manifest is not None:
        records = load_bridgedata_manifest_jsonl(manifest)
    elif config["dataset"].get("use_fake_manifest_if_missing", False):
        fake_cfg = config["dataset"].get("fake_manifest", {})
        records = make_fake_bridgedata_manifest_records(
            num_trajectories=int(fake_cfg.get("num_trajectories", 3)),
            num_frames=int(fake_cfg.get("num_frames", 40)),
        )
        used_fake_manifest = True
    else:
        raise FileNotFoundError("No BridgeData V2 tiny-subset manifest found and fake manifest is disabled.")

    spec = window_spec_from_dict(config["window"])
    windows = build_bridgedata_windows_from_manifest(records, spec)
    output_records = config["output_records"]
    manifest_path = write_window_manifest_jsonl(output_records["window_manifest_jsonl"], windows)
    trajectory_summary = summarize_manifest_window_build(records, windows, spec)
    summary = {
        "stage": config["stage"],
        "dataset_name": "BridgeData V2",
        "local_subset_exists": inspection["local_subset_exists"],
        "manifest_found": inspection["manifest_found"],
        "used_fake_manifest": used_fake_manifest,
        "full_download_performed": False,
        "large_download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "num_trajectories": trajectory_summary["num_trajectories"],
        "num_valid_trajectories": trajectory_summary["num_valid_trajectories"],
        "num_skipped_trajectories": trajectory_summary["num_skipped_trajectories"],
        "skipped_trajectories": trajectory_summary["skipped_trajectories"],
        "num_windows": trajectory_summary["num_windows"],
        "context_len": spec.context_len,
        "current_len": spec.current_len,
        "future_len": spec.future_len,
        "use_action_as_input": False,
        "use_language_as_input": False,
        "use_goal_image_as_input": False,
        "window_manifest_jsonl": str(manifest_path),
        "trajectory_summary_json": output_records["trajectory_summary_json"],
        "metadata_policy": config["metadata_policy"],
    }
    summary["sanity_gate_pass"] = bool(
        not summary["full_download_performed"]
        and not summary["large_download_performed"]
        and not summary["training_performed"]
        and not summary["token_extraction_performed"]
        and not summary["importance_generation_performed"]
        and not summary["use_action_as_input"]
        and not summary["use_language_as_input"]
        and not summary["use_goal_image_as_input"]
        and summary["num_windows"] > 0
    )
    Path(output_records["trajectory_summary_json"]).write_text(json.dumps(trajectory_summary, indent=2), encoding="utf-8")
    Path(output_records["builder_summary_json"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    Path(output_records["builder_summary_md"]).write_text(render_builder_summary_markdown(summary), encoding="utf-8")
    return summary


def render_builder_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# BridgeData V2 Window Builder Summary",
        "",
        f"- local_subset_exists: `{str(summary['local_subset_exists']).lower()}`",
        f"- manifest_found: `{str(summary['manifest_found']).lower()}`",
        f"- used_fake_manifest: `{str(summary['used_fake_manifest']).lower()}`",
        f"- num_trajectories: `{summary['num_trajectories']}`",
        f"- num_valid_trajectories: `{summary['num_valid_trajectories']}`",
        f"- num_skipped_trajectories: `{summary['num_skipped_trajectories']}`",
        f"- num_windows: `{summary['num_windows']}`",
        f"- context/current/future: `{summary['context_len']}/{summary['current_len']}/{summary['future_len']}`",
        f"- sanity_gate_pass: `{str(summary['sanity_gate_pass']).lower()}`",
        "",
        "## Boundaries",
        "",
        "- no full dataset download",
        "- no training",
        "- no token extraction",
        "- no importance generation",
        "- action/language/goal image saved as metadata only",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = build_from_config(args.config)
    print(json.dumps({"num_windows": summary["num_windows"], "used_fake_manifest": summary["used_fake_manifest"]}, indent=2))


if __name__ == "__main__":
    main()
