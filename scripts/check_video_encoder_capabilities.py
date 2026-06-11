"""Check Step 9B frozen video encoder capabilities without downloading weights."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from encoders.frozen_video_encoder import build_frozen_video_encoder
from scripts.check_resource_limits import collect_resource_summary


CAPABILITY_JSON = PROJECT_ROOT / "docs" / "frozen_video_encoder_capabilities.json"
CAPABILITY_REPORT = PROJECT_ROOT / "docs" / "FROZEN_VIDEO_ENCODER_CAPABILITY_REPORT.md"


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _cache_status() -> list[dict[str, Any]]:
    candidates = [
        Path.home() / ".cache" / "huggingface",
        Path("/home/ubuntu22/.cache/huggingface"),
        PROJECT_ROOT / "model_cache",
    ]
    paths: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            paths.append(path)
            seen.add(key)
    status = []
    for path in paths:
        status.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "num_entries": len(list(path.iterdir())) if path.exists() and path.is_dir() else 0,
            }
        )
    return status


def collect_video_encoder_capabilities() -> dict[str, Any]:
    resource = collect_resource_summary(PROJECT_ROOT)
    videomae = build_frozen_video_encoder(
        {
            "name": "videomae",
            "model_name_or_path": None,
            "allow_download": False,
            "local_files_only": True,
            "device": "cuda_if_available",
        }
    )
    vjepa = build_frozen_video_encoder({"name": "vjepa"})
    return {
        "python": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "resource": resource,
        "transformers_available": _module_available("transformers"),
        "torchvision_available": _module_available("torchvision"),
        "huggingface_cache": _cache_status(),
        "videomae": videomae.availability,
        "vjepa": vjepa.availability,
        "download_policy": {
            "allow_download_default": False,
            "local_files_only_default": True,
            "notes": "Step 9B capability checks never download model weights.",
        },
    }


def write_capability_report(summary: dict[str, Any]) -> None:
    CAPABILITY_JSON.parent.mkdir(parents=True, exist_ok=True)
    CAPABILITY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    resource = summary["resource"]
    lines = [
        "# Frozen Video Encoder Capability Report",
        "",
        "Command: `python scripts/check_video_encoder_capabilities.py`",
        "",
        "## Runtime",
        "",
        f"- Python: `{summary['python']}`",
        f"- torch: `{summary['torch_version']}`",
        f"- CUDA available: `{summary['cuda_available']}`",
        f"- GPU: `{resource['gpu_name']}`",
        f"- GPU memory total/free GiB: `{resource['gpu_memory_total_gib']} / {resource['gpu_memory_free_gib']}`",
        f"- RAM total/available GiB: `{resource['ram_total_gib']} / {resource['ram_available_gib']}`",
        "",
        "## Optional Dependencies",
        "",
        f"- transformers available: `{summary['transformers_available']}`",
        f"- torchvision available: `{summary['torchvision_available']}`",
        "",
        "## Local Model Cache",
        "",
    ]
    for item in summary["huggingface_cache"]:
        lines.append(f"- `{item['path']}` exists=`{item['exists']}` entries=`{item['num_entries']}`")
    lines.extend(
        [
            "",
            "## Encoder Availability",
            "",
            f"- VideoMAE: `{summary['videomae']['available']}`; reason: `{summary['videomae']['reason']}`",
            f"- V-JEPA: `{summary['vjepa']['available']}`; reason: `{summary['vjepa']['reason']}`",
            "",
            "## Download Policy",
            "",
            "- `allow_download` defaults to `false`.",
            "- `local_files_only` defaults to `true`.",
            "- This check does not download model weights.",
            "",
            "```json",
            json.dumps(summary, indent=2),
            "```",
            "",
        ]
    )
    CAPABILITY_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary = collect_video_encoder_capabilities()
    write_capability_report(summary)
    print(json.dumps(summary, indent=2))
    print(f"FROZEN_VIDEO_ENCODER_CAPABILITY_JSON = {CAPABILITY_JSON}")
    print(f"FROZEN_VIDEO_ENCODER_CAPABILITY_REPORT = {CAPABILITY_REPORT}")


if __name__ == "__main__":
    main()
