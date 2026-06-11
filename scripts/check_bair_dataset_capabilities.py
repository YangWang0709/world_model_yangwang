"""Check whether the environment can download BAIR Robot Pushing small."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bair_tfds_utils import check_tfds_available


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bair_robot_pushing_small.yaml"
REPORT_PATH = PROJECT_ROOT / "docs" / "BAIR_CAPABILITY_REPORT.md"
JSON_PATH = PROJECT_ROOT / "docs" / "bair_capabilities.json"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def _ram_available_gib() -> float:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                kib = float(line.split()[1])
                return kib / (1024.0**2)
    return 0.0


def _proxy_env_summary() -> dict[str, str]:
    proxy_keys = [key for key in os.environ if "proxy" in key.lower()]
    return {key: "set" for key in sorted(proxy_keys)}


def _partial_tfds_summary(data_dir: Path) -> dict[str, Any]:
    if not data_dir.exists():
        return {
            "bair_data_dir_exists": False,
            "partial_tfds_download_detected": False,
            "top_level_entries": [],
        }
    entries = sorted(child.name for child in data_dir.iterdir())[:50]
    partial = any(
        "incomplete" in name.lower()
        or name.endswith(".tmp")
        or name.endswith(".lock")
        or name in {"downloads", "manual_dir"}
        for name in entries
    )
    return {
        "bair_data_dir_exists": True,
        "partial_tfds_download_detected": partial,
        "top_level_entries": entries,
    }


def build_capability_summary(
    config_path: str | Path = DEFAULT_CONFIG,
    write_outputs: bool = False,
) -> dict[str, Any]:
    """Build a capability summary without installing dependencies or downloading data."""

    config = _load_yaml(config_path)
    project_root = Path(config.get("project_root", PROJECT_ROOT))
    dataset_cfg = config.get("dataset", {})
    limits = config.get("resource_limits", {})
    data_dir = Path(dataset_cfg.get("data_dir", project_root / "data" / "bair_robot_pushing_small_tfds"))
    min_disk_required = float(limits.get("min_disk_free_gib", 80))
    disk_free_gib = shutil.disk_usage(project_root).free / (1024.0**3)
    tfds_summary = check_tfds_available()
    download_allowed = bool(dataset_cfg.get("download", False))

    blocking_reason = None
    if disk_free_gib < min_disk_required:
        blocking_reason = f"disk free {disk_free_gib:.2f} GiB is below required {min_disk_required:.2f} GiB"
    elif not tfds_summary["tensorflow_datasets_available"]:
        blocking_reason = "tensorflow_datasets is not available"
    elif not tfds_summary.get("tfds_builder_available", False):
        blocking_reason = tfds_summary.get("tfds_builder_error") or "BAIR TFDS builder is not available"
    elif not download_allowed:
        blocking_reason = "dataset.download is not true in config"

    partial_summary = _partial_tfds_summary(data_dir)
    summary: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_available": "torch" in sys.modules or _module_available("torch"),
        "numpy_available": "numpy" in sys.modules or _module_available("numpy"),
        "tensorflow_available": bool(tfds_summary["tensorflow_available"]),
        "tensorflow_datasets_available": bool(tfds_summary["tensorflow_datasets_available"]),
        "tensorflow_version": tfds_summary.get("tensorflow_version"),
        "tensorflow_datasets_version": tfds_summary.get("tensorflow_datasets_version"),
        "tfds_builder_available": bool(tfds_summary.get("tfds_builder_available", False)),
        "tfds_builder_error": tfds_summary.get("tfds_builder_error"),
        "disk_free_gib": disk_free_gib,
        "ram_available_gib": _ram_available_gib(),
        "download_allowed": download_allowed,
        "min_disk_required_gib": min_disk_required,
        "can_attempt_download": blocking_reason is None,
        "blocking_reason": blocking_reason,
        "proxy_env": _proxy_env_summary(),
        "data_dir": str(data_dir),
        **partial_summary,
    }
    if write_outputs:
        write_capability_outputs(summary, REPORT_PATH, JSON_PATH)
    return summary


def _module_available(module_name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module_name) is not None


def write_capability_outputs(summary: dict[str, Any], report_path: Path, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# BAIR Capability Report",
        "",
        f"- tensorflow_available: `{str(summary['tensorflow_available']).lower()}`",
        f"- tensorflow_datasets_available: `{str(summary['tensorflow_datasets_available']).lower()}`",
        f"- tfds_builder_available: `{str(summary['tfds_builder_available']).lower()}`",
        f"- disk_free_gib: `{summary['disk_free_gib']}`",
        f"- ram_available_gib: `{summary['ram_available_gib']}`",
        f"- min_disk_required_gib: `{summary['min_disk_required_gib']}`",
        f"- download_allowed: `{str(summary['download_allowed']).lower()}`",
        f"- can_attempt_download: `{str(summary['can_attempt_download']).lower()}`",
        f"- blocking_reason: `{summary['blocking_reason']}`",
        f"- data_dir: `{summary['data_dir']}`",
        f"- bair_data_dir_exists: `{str(summary['bair_data_dir_exists']).lower()}`",
        f"- partial_tfds_download_detected: `{str(summary['partial_tfds_download_detected']).lower()}`",
        "",
        "Proxy environment variables are reported by name only; values are intentionally redacted.",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_capability_summary(args.config, write_outputs=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"BAIR_CAN_ATTEMPT_DOWNLOAD = {str(summary['can_attempt_download']).lower()}")


if __name__ == "__main__":
    main()
