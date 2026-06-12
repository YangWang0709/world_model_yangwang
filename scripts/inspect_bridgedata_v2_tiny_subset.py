"""Inspect a user-provided BridgeData V2 tiny subset without reading media."""

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

from data.bridgedata_v2_format_inspector import (
    find_first_existing_manifest,
    find_first_existing_subset_dir,
    inspect_bridgedata_v2_subset,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tiny_window_builder_step20.yaml"


def _resolve_run_dir(config: dict[str, Any]) -> Path:
    return Path(config["output_records"]["output_root"])


def inspect_from_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = _resolve_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset = config["dataset"]
    manifest = find_first_existing_manifest(dataset.get("manifest_candidates", []))
    if manifest is not None:
        inspection = inspect_bridgedata_v2_subset(manifest.parent)
    else:
        subset_dir = find_first_existing_subset_dir(dataset.get("local_subset_candidates", []))
        inspection = inspect_bridgedata_v2_subset(subset_dir or dataset.get("local_subset_candidates", [None])[0])
    inspection.update(
        {
            "stage": config["stage"],
            "allow_missing_local_subset": bool(dataset.get("allow_missing_local_subset", False)),
            "fake_manifest_available": bool(dataset.get("use_fake_manifest_if_missing", False)),
            "full_download_performed": False,
            "large_download_performed": False,
            "training_performed": False,
            "token_extraction_performed": False,
            "importance_generation_performed": False,
        }
    )
    (run_dir / "bridgedata_v2_subset_inspection.json").write_text(json.dumps(inspection, indent=2), encoding="utf-8")
    (run_dir / "bridgedata_v2_subset_inspection.md").write_text(render_inspection_markdown(inspection), encoding="utf-8")
    return inspection


def render_inspection_markdown(inspection: dict[str, Any]) -> str:
    lines = [
        "# BridgeData V2 Tiny Subset Inspection",
        "",
        f"- local_subset_exists: `{str(inspection['local_subset_exists']).lower()}`",
        f"- manifest_found: `{str(inspection['manifest_found']).lower()}`",
        f"- detected_layout: `{inspection['detected_layout']}`",
        f"- trajectory_count_estimate: `{inspection['trajectory_count_estimate']}`",
        f"- fake_manifest_available: `{str(inspection['fake_manifest_available']).lower()}`",
        "- no_download: `true`",
        "- training_performed: `false`",
        "- token_extraction_performed: `false`",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in inspection.get("warnings", []))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    inspection = inspect_from_config(args.config)
    print(json.dumps({"local_subset_exists": inspection["local_subset_exists"], "manifest_found": inspection["manifest_found"]}, indent=2))


if __name__ == "__main__":
    main()
