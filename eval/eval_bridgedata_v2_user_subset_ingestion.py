"""Evaluate and report Step22 user-provided BridgeData V2 subset ingestion."""

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

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_user_subset_ingestion_step22.yaml"


def evaluate_user_subset_ingestion(config_path: Path = DEFAULT_CONFIG, docs_dir: Path | None = None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = config["output"]
    inspection = _read_json(output["subset_inspection_json"])
    manifest = _read_json(output["manifest_summary_json"])
    validation = _read_json(output["validation_summary_json"])
    window = _read_json(output["window_builder_summary_json"])
    pending = bool(window.get("pending_user_data", False))
    validated = bool(window.get("real_format_validated", False))
    summary = {
        "stage": "bridgedata_v2_user_subset_ingestion_step22",
        "safety_gate_pass": bool(
            window.get("safety_gate_pass", False)
            and not window.get("download_performed", False)
            and not window.get("training_performed", False)
            and not window.get("token_extraction_performed", False)
            and not window.get("importance_generation_performed", False)
        ),
        "user_subset_exists": bool(window.get("user_subset_exists", False)),
        "pending_user_data": pending,
        "real_format_validated": validated,
        "manifest_exists": bool(window.get("manifest_exists", False)),
        "generated_manifest_exists": bool(window.get("generated_manifest_exists", False)),
        "user_subset_window_manifest_exists": bool(window.get("user_window_manifest_exists", False)),
        "num_trajectories": int(window.get("num_manifest_records", 0)),
        "num_valid_trajectories": int(window.get("num_valid_trajectories", 0)),
        "num_skipped_trajectories": int(window.get("num_skipped_trajectories", 0)),
        "num_windows": int(window.get("num_windows", 0)),
        "frame_count_min": window.get("frame_count_min"),
        "frame_count_mean": window.get("frame_count_mean"),
        "frame_count_max": window.get("frame_count_max"),
        "metadata_fields": {
            "images_likely": window.get("has_images_likely"),
            "actions_likely": window.get("has_actions_likely"),
            "language_likely": window.get("has_language_likely"),
            "goal_image_likely": window.get("has_goal_image_likely"),
        },
        "download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "use_action_as_input": False,
        "use_language_as_input": False,
        "use_goal_image_as_input": False,
        "recommended_step23": "BridgeData V2 user-subset token extraction dry-run"
        if validated
        else "User prepares tiny subset according to checklist",
        "inspection": inspection,
        "manifest_summary": manifest,
        "validation": validation,
        "window_summary": window,
    }
    eval_json = Path(output["eval_json"])
    eval_md = Path(output["eval_md"])
    docs_root = docs_dir if docs_dir is not None else PROJECT_ROOT / "docs"
    docs_root.mkdir(parents=True, exist_ok=True)
    docs_report = docs_root / "BRIDGEDATA_V2_USER_SUBSET_INGESTION_REPORT.md"
    eval_json.parent.mkdir(parents=True, exist_ok=True)
    eval_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    rendered = render_user_subset_ingestion_report(summary)
    eval_md.write_text(rendered, encoding="utf-8")
    docs_report.write_text(rendered, encoding="utf-8")
    return summary


def render_user_subset_ingestion_report(summary: dict[str, Any]) -> str:
    if summary["pending_user_data"]:
        status_lines = [
            "User subset is not available yet.",
            "No real format validation was performed.",
            "Safety gate passed.",
            "Next step: user should prepare subset according to checklist.",
        ]
    elif summary["real_format_validated"]:
        status_lines = [
            "User subset exists.",
            "Real format validated.",
            "Window manifest generated.",
            "Next step: BridgeData V2 user-subset token extraction dry-run.",
        ]
    else:
        status_lines = [
            "User subset exists but was not validated.",
            "Window manifest was not generated.",
            "Safety gate passed for non-training ingestion checks.",
            "Next step: fix the local subset format according to checklist.",
        ]
    lines = [
        "# BridgeData V2 User Subset Ingestion Report",
        "",
        *status_lines,
        "",
        "## Summary",
        "",
        f"- safety_gate_pass: `{str(summary['safety_gate_pass']).lower()}`",
        f"- user_subset_exists: `{str(summary['user_subset_exists']).lower()}`",
        f"- pending_user_data: `{str(summary['pending_user_data']).lower()}`",
        f"- real_format_validated: `{str(summary['real_format_validated']).lower()}`",
        f"- user_subset_window_manifest_exists: `{str(summary['user_subset_window_manifest_exists']).lower()}`",
        f"- num_trajectories: `{summary['num_trajectories']}`",
        f"- num_valid_trajectories: `{summary['num_valid_trajectories']}`",
        f"- num_skipped_trajectories: `{summary['num_skipped_trajectories']}`",
        f"- num_windows: `{summary['num_windows']}`",
        f"- frame count min/mean/max: `{summary['frame_count_min']}/{summary['frame_count_mean']}/{summary['frame_count_max']}`",
        "",
        "## Metadata Fields",
        "",
        f"- images likely: `{summary['metadata_fields']['images_likely']}`",
        f"- actions likely: `{summary['metadata_fields']['actions_likely']}`",
        f"- language likely: `{summary['metadata_fields']['language_likely']}`",
        f"- goal image likely: `{summary['metadata_fields']['goal_image_likely']}`",
        "",
        "## Boundaries",
        "",
        "- no download",
        "- no training",
        "- no token extraction",
        "- no importance generation",
        "- action, language, and goal image remain metadata only",
        "",
        "## Recommended Step23",
        "",
        f"`{summary['recommended_step23']}`",
        "",
    ]
    return "\n".join(lines)


def _read_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(evaluate_user_subset_ingestion(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
