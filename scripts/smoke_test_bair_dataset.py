"""End-to-end BAIR dataset smoke for download, subset export, and loader checks."""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bair_dataset import BAIRRobotPushingDataset
from data.bair_subset_export import export_bair_subset
from scripts.check_bair_dataset_capabilities import build_capability_summary
from scripts.download_bair_robot_pushing_small import run_download


DOWNLOAD_CONFIG = PROJECT_ROOT / "configs" / "bair_robot_pushing_small.yaml"
SUBSET_CONFIG = PROJECT_ROOT / "configs" / "bair_robot_pushing_subset.yaml"
REPORT_PATH = PROJECT_ROOT / "docs" / "BAIR_DATASET_SMOKE_REPORT.md"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def _disk_free_gib(path: str | Path) -> float:
    return shutil.disk_usage(path).free / (1024.0**3)


def _metadata_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _subset_ready(output_root: Path) -> bool:
    return _metadata_count(output_root / "train" / "metadata.jsonl") > 0 and _metadata_count(
        output_root / "test" / "metadata.jsonl"
    ) > 0


def _shape(value: Any) -> list[int] | None:
    if hasattr(value, "shape"):
        return [int(dim) for dim in value.shape]
    return None


def _finite_range(value: Any) -> tuple[float | None, float | None, bool]:
    if value is None:
        return None, None, False
    min_value = float(value.min())
    max_value = float(value.max())
    return min_value, max_value, math.isfinite(min_value) and math.isfinite(max_value)


def _write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# BAIR Dataset Smoke Report",
        "",
        "Command: `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_bair_dataset.py`",
        "",
        f"- BAIR_DATASET_SMOKE_PASS: `{str(summary.get('smoke_pass', False)).lower()}`",
        f"- blocking_reason: `{summary.get('blocking_reason')}`",
        f"- download_status: `{summary.get('download_status')}`",
        f"- subset_export_status: `{summary.get('subset_export_status')}`",
        f"- train subset size: `{summary.get('train_subset_size')}`",
        f"- test subset size: `{summary.get('test_subset_size')}`",
        f"- past shape: `{summary.get('past_shape')}`",
        f"- future shape: `{summary.get('future_shape')}`",
        f"- actions shape: `{summary.get('actions_shape')}`",
        f"- value range: `{summary.get('value_range')}`",
        f"- disk_free_before_gib: `{summary.get('disk_free_before_gib')}`",
        f"- disk_free_after_gib: `{summary.get('disk_free_after_gib')}`",
        f"- elapsed_time_sec: `{summary.get('elapsed_time_sec')}`",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        f"BAIR_DATASET_SMOKE_PASS = {str(summary.get('smoke_pass', False)).lower()}",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke() -> dict[str, Any]:
    start = time.time()
    download_config = _load_yaml(DOWNLOAD_CONFIG)
    subset_config = _load_yaml(SUBSET_CONFIG)
    project_root = Path(download_config.get("project_root", PROJECT_ROOT))
    data_dir = Path(download_config["dataset"]["data_dir"])
    output_root = Path(subset_config["subset"]["output_root"])
    disk_before = _disk_free_gib(project_root)
    capability = build_capability_summary(DOWNLOAD_CONFIG, write_outputs=True)
    summary: dict[str, Any] = {
        "smoke_pass": False,
        "blocking_reason": None,
        "capability_summary": capability,
        "download_status": "not_run",
        "download_summary": None,
        "subset_export_status": "not_run",
        "subset_export_summary": None,
        "data_dir": str(data_dir),
        "subset_output_root": str(output_root),
        "train_subset_size": 0,
        "test_subset_size": 0,
        "past_shape": None,
        "future_shape": None,
        "actions_shape": None,
        "endeffector_pos_shape": None,
        "value_range": None,
        "disk_free_before_gib": disk_before,
        "disk_free_after_gib": disk_before,
        "elapsed_time_sec": 0.0,
    }

    if not data_dir.exists() or not any(data_dir.iterdir()):
        download_summary = run_download(DOWNLOAD_CONFIG)
        summary["download_summary"] = download_summary
        summary["download_status"] = "success" if download_summary.get("download_success") else "blocked"
        if not download_summary.get("download_success"):
            summary["blocking_reason"] = download_summary.get("blocking_reason")
    else:
        summary["download_status"] = "reused_existing"

    if not _subset_ready(output_root):
        export_summary = export_bair_subset(SUBSET_CONFIG)
        summary["subset_export_summary"] = export_summary
        summary["subset_export_status"] = "success" if export_summary.get("export_success") else "blocked"
        if not export_summary.get("export_success") and summary["blocking_reason"] is None:
            summary["blocking_reason"] = export_summary.get("blocking_reason")
    else:
        summary["subset_export_status"] = "reused_existing"

    if _subset_ready(output_root):
        try:
            train_dataset = BAIRRobotPushingDataset(
                root=output_root,
                split="train",
                past_len=int(subset_config["subset"].get("past_len", 4)),
                future_len=int(subset_config["subset"].get("future_len", 4)),
                image_size=int(subset_config["subset"].get("image_size", 224)),
            )
            test_dataset = BAIRRobotPushingDataset(
                root=output_root,
                split="test",
                past_len=int(subset_config["subset"].get("past_len", 4)),
                future_len=int(subset_config["subset"].get("future_len", 4)),
                image_size=int(subset_config["subset"].get("image_size", 224)),
            )
            sample = train_dataset[0]
            past_min, past_max, past_finite = _finite_range(sample["past_video"])
            future_min, future_max, future_finite = _finite_range(sample["future_video"])
            summary.update(
                {
                    "train_subset_size": len(train_dataset),
                    "test_subset_size": len(test_dataset),
                    "past_shape": _shape(sample["past_video"]),
                    "future_shape": _shape(sample["future_video"]),
                    "actions_shape": _shape(sample.get("actions")),
                    "endeffector_pos_shape": _shape(sample.get("endeffector_pos")),
                    "sample_id": sample.get("sample_id"),
                    "value_range": {
                        "past_min": past_min,
                        "past_max": past_max,
                        "future_min": future_min,
                        "future_max": future_max,
                    },
                }
            )
            summary["smoke_pass"] = (
                summary["past_shape"] == [4, 3, 224, 224]
                and summary["future_shape"] == [4, 3, 224, 224]
                and past_finite
                and future_finite
                and past_min is not None
                and future_min is not None
                and past_min >= -1e-6
                and future_min >= -1e-6
                and past_max is not None
                and future_max is not None
                and past_max <= 1.0 + 1e-6
                and future_max <= 1.0 + 1e-6
                and bool(summary.get("sample_id"))
            )
            if not summary["smoke_pass"] and summary["blocking_reason"] is None:
                summary["blocking_reason"] = "loader sample checks failed"
        except Exception as exc:
            summary["blocking_reason"] = str(exc)

    summary["train_subset_size"] = summary["train_subset_size"] or _metadata_count(output_root / "train" / "metadata.jsonl")
    summary["test_subset_size"] = summary["test_subset_size"] or _metadata_count(output_root / "test" / "metadata.jsonl")
    summary["disk_free_after_gib"] = _disk_free_gib(project_root)
    summary["elapsed_time_sec"] = time.time() - start
    _write_report(summary)
    return summary


def main() -> None:
    summary = run_smoke()
    print(REPORT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
