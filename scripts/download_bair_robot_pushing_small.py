"""Download BAIR Robot Pushing small through TFDS when the environment is ready."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bair_tfds_utils import download_bair_dataset
from scripts.check_bair_dataset_capabilities import build_capability_summary


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bair_robot_pushing_small.yaml"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def _disk_free_gib(path: str | Path) -> float:
    return shutil.disk_usage(path).free / (1024.0**3)


def _write_download_report(report_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# BAIR Download Report",
        "",
        f"- download_success: `{str(summary.get('download_success', False)).lower()}`",
        f"- blocking_reason: `{summary.get('blocking_reason')}`",
        f"- data_dir: `{summary.get('data_dir')}`",
        f"- disk_free_before_gib: `{summary.get('disk_free_before_gib')}`",
        f"- disk_free_after_gib: `{summary.get('disk_free_after_gib')}`",
        f"- elapsed_time_sec: `{summary.get('elapsed_time_sec')}`",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        f"BAIR_DOWNLOAD_PASS = {str(summary.get('download_success', False)).lower()}",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_download(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_yaml(config_path)
    project_root = Path(config.get("project_root", PROJECT_ROOT))
    dataset_cfg = config["dataset"]
    limits = config.get("resource_limits", {})
    report_path = Path(config.get("output", {}).get("report_path", PROJECT_ROOT / "docs" / "BAIR_DOWNLOAD_REPORT.md"))
    max_retries = max(1, int(limits.get("max_retries", 1)))
    data_dir = Path(dataset_cfg["data_dir"])
    start = time.time()
    disk_before = _disk_free_gib(project_root)
    capability = build_capability_summary(config_path, write_outputs=True)
    summary: dict[str, Any] = {
        "download_success": False,
        "blocking_reason": capability.get("blocking_reason"),
        "data_dir": str(data_dir),
        "disk_free_before_gib": disk_before,
        "disk_free_after_gib": disk_before,
        "elapsed_time_sec": 0.0,
        "attempts": 0,
        "capability_summary": capability,
        "dataset_info": None,
    }
    if not capability["can_attempt_download"]:
        summary["elapsed_time_sec"] = time.time() - start
        _write_download_report(report_path, summary)
        return summary
    if not bool(dataset_cfg.get("download", False)):
        summary["blocking_reason"] = "dataset.download is not true in config"
        summary["elapsed_time_sec"] = time.time() - start
        _write_download_report(report_path, summary)
        return summary

    last_error: str | None = None
    for attempt in range(1, max_retries + 1):
        summary["attempts"] = attempt
        try:
            result = download_bair_dataset(
                data_dir=str(data_dir),
                download=True,
                tfds_name=str(dataset_cfg.get("tfds_name", "bair_robot_pushing_small")),
                tfds_version=str(dataset_cfg.get("tfds_version", "2.0.0")),
            )
            summary["download_success"] = True
            summary["blocking_reason"] = None
            summary["dataset_info"] = result.get("dataset_info")
            break
        except Exception as exc:  # pragma: no cover - network/dataset environment dependent
            last_error = str(exc)
            summary["blocking_reason"] = last_error
            break

    summary["disk_free_after_gib"] = _disk_free_gib(project_root)
    summary["elapsed_time_sec"] = time.time() - start
    if last_error and not summary["download_success"]:
        summary["blocking_reason"] = last_error
    _write_download_report(report_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_download(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"BAIR_DOWNLOAD_PASS = {str(summary['download_success']).lower()}")


if __name__ == "__main__":
    main()
