"""Collect conservative resource guidance for Step 9A real-video smoke tests."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_PATH = PROJECT_ROOT / "docs" / "REAL_VIDEO_RESOURCE_REPORT.md"


def _bytes_to_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024**3), 3)


def _memory_summary() -> dict[str, Any]:
    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        return {
            "ram_total_gib": _bytes_to_gib(memory.total),
            "ram_available_gib": _bytes_to_gib(memory.available),
        }
    except ImportError:
        meminfo: dict[str, int] = {}
        meminfo_path = Path("/proc/meminfo")
        if meminfo_path.exists():
            for line in meminfo_path.read_text(encoding="utf-8").splitlines():
                key, value = line.split(":", 1)
                meminfo[key] = int(value.strip().split()[0]) * 1024
        return {
            "ram_total_gib": _bytes_to_gib(meminfo.get("MemTotal")),
            "ram_available_gib": _bytes_to_gib(meminfo.get("MemAvailable")),
        }


def collect_resource_summary(project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root)
    disk = shutil.disk_usage(root)
    memory = _memory_summary()
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = None
    gpu_memory_total_gib = None
    gpu_memory_free_gib = None
    if cuda_available:
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(device)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            gpu_memory_total_gib = _bytes_to_gib(total_bytes)
            gpu_memory_free_gib = _bytes_to_gib(free_bytes)
        except RuntimeError:
            properties = torch.cuda.get_device_properties(device)
            gpu_memory_total_gib = _bytes_to_gib(properties.total_memory)

    warnings: list[str] = []
    ram_available_gib = memory.get("ram_available_gib")
    disk_free_gib = _bytes_to_gib(disk.free)
    if ram_available_gib is not None and ram_available_gib < 8:
        warnings.append("RAM available is below 8 GiB; keep batch_size=1 and num_workers=0")
    if disk_free_gib is not None and disk_free_gib < 50:
        warnings.append("Disk free space is below 50 GiB; avoid preparing additional video subsets")
    if not cuda_available:
        warnings.append("CUDA is unavailable; Step 9A dummy extraction can still run on CPU")

    return {
        "cpu_count": os.cpu_count(),
        **memory,
        "disk_free_gib": disk_free_gib,
        "disk_total_gib": _bytes_to_gib(disk.total),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_memory_total_gib": gpu_memory_total_gib,
        "gpu_memory_free_gib": gpu_memory_free_gib,
        "recommended": {
            "batch_size": 2,
            "num_workers": 0,
            "max_samples": 100,
            "clip_len": 8,
            "image_size": 224,
        },
        "warnings": warnings,
    }


def write_resource_report(summary: dict[str, Any], report_path: str | Path = REPORT_PATH) -> Path:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Real Video Resource Report",
        "",
        "Command: `python scripts/check_resource_limits.py`",
        "",
        "## Summary",
        "",
        f"- CPU count: `{summary['cpu_count']}`",
        f"- RAM total GiB: `{summary['ram_total_gib']}`",
        f"- RAM available GiB: `{summary['ram_available_gib']}`",
        f"- Disk total GiB: `{summary['disk_total_gib']}`",
        f"- Disk free GiB: `{summary['disk_free_gib']}`",
        f"- CUDA available: `{summary['cuda_available']}`",
        f"- GPU name: `{summary['gpu_name']}`",
        f"- GPU memory total GiB: `{summary['gpu_memory_total_gib']}`",
        f"- GPU memory free GiB: `{summary['gpu_memory_free_gib']}`",
        "",
        "## Conservative Step 9A Settings",
        "",
        "- `batch_size <= 2`",
        "- `num_workers = 0` by default",
        "- `max_samples <= 100`",
        "- `clip_len <= 8`",
        "- `image_size <= 224`",
        "",
        "## Warnings",
        "",
    ]
    if summary["warnings"]:
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    else:
        lines.append("- none")
    lines.extend(["", "```json", json.dumps(summary, indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    summary = collect_resource_summary(PROJECT_ROOT)
    report_path = write_resource_report(summary)
    print(json.dumps(summary, indent=2))
    print(f"REAL_VIDEO_RESOURCE_REPORT_WRITTEN = {report_path}")


if __name__ == "__main__":
    main()
