"""Download or verify one conservative VideoMAE checkpoint for Step 9C."""

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

DEFAULT_MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "model_cache" / "huggingface"
REPORT_PATH = PROJECT_ROOT / "docs" / "VIDEOMAE_DOWNLOAD_REPORT.md"


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--allow-download", type=_str_to_bool, default=False)
    parser.add_argument("--local-files-only", type=_str_to_bool, default=False)
    return parser.parse_args()


def _bytes_to_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024**3), 3)


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _disk_summary(path: Path) -> dict[str, float | None]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return {
        "total_gib": _bytes_to_gib(usage.total),
        "used_gib": _bytes_to_gib(usage.used),
        "free_gib": _bytes_to_gib(usage.free),
        "cache_size_gib": _bytes_to_gib(_dir_size_bytes(path)),
    }


def _cache_file_summary(cache_dir: Path, limit: int = 12) -> dict[str, Any]:
    files = sorted([item for item in cache_dir.rglob("*") if item.is_file()], key=lambda p: str(p))
    selected = []
    for item in files[:limit]:
        selected.append(
            {
                "path": str(item.relative_to(cache_dir)),
                "size_mib": round(item.stat().st_size / (1024**2), 3),
            }
        )
    return {
        "file_count": len(files),
        "total_size_gib": _bytes_to_gib(sum(item.stat().st_size for item in files)),
        "shown_files": selected,
        "truncated": len(files) > limit,
    }


def write_download_report(summary: dict[str, Any], report_path: str | Path = REPORT_PATH) -> Path:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VideoMAE Download Report",
        "",
        "Command: `python scripts/download_videomae_checkpoint.py --model-name <checkpoint-or-repo> --allow-download <true|false>`",
        "",
        f"- model_name: `{summary['model_name']}`",
        f"- cache_dir: `{summary['cache_dir']}`",
        f"- source: `{summary['source']}`",
        f"- allow_download: `{summary['allow_download']}`",
        f"- local_files_only: `{summary['local_files_only']}`",
        f"- download_success: `{summary['download_success']}`",
        f"- elapsed_time_sec: `{summary['elapsed_time_sec']}`",
        f"- snapshot_path: `{summary.get('snapshot_path')}`",
        f"- error: `{summary.get('error')}`",
        "",
        "## Disk Usage",
        "",
        f"- before: `{summary['disk_before']}`",
        f"- after: `{summary['disk_after']}`",
        "",
        "## Cached Files Summary",
        "",
        "```json",
        json.dumps(summary["cache_files"], indent=2),
        "```",
        "",
        "## Boundary Confirmation",
        "",
        "- one VideoMAE checkpoint was allowed for Step 9C",
        "- weights are stored under ignored `model_cache/`",
        "- no model weight files should be committed to git",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def download_videomae_checkpoint(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    allow_download: bool = False,
    local_files_only: bool = False,
) -> dict[str, Any]:
    cache_path = Path(cache_dir)
    if not allow_download:
        local_files_only = True
    disk_before = _disk_summary(cache_path)
    start_time = time.perf_counter()
    snapshot_path: str | None = None
    error: str | None = None
    success = False
    model_path = Path(model_name)
    local_files_summary: dict[str, Any] | None = None

    if model_path.exists():
        required = ["config.json"]
        weight_candidates = ["model.safetensors", "pytorch_model.bin"]
        missing = [name for name in required if not (model_path / name).exists()]
        has_weight = any((model_path / name).exists() for name in weight_candidates)
        if missing or not has_weight:
            error = (
                f"local checkpoint path exists but is incomplete: missing={missing}, "
                f"has_weight={has_weight}"
            )
        else:
            snapshot_path = str(model_path)
            success = True
            local_files_summary = _cache_file_summary(model_path)
    else:
        try:
            from huggingface_hub import snapshot_download  # type: ignore

            snapshot_path = snapshot_download(
                repo_id=model_name,
                cache_dir=str(cache_path),
                local_files_only=local_files_only,
                max_workers=1,
                allow_patterns=[
                    "config.json",
                    "preprocessor_config.json",
                    "image_processor_config.json",
                    "model.safetensors",
                    "README.md",
                ],
            )
            success = True
        except Exception as exc:  # pragma: no cover - depends on network/cache state
            error = str(exc)

    summary = {
        "model_name": model_name,
        "cache_dir": str(cache_path),
        "source": "local_path" if model_path.exists() else "huggingface_hub",
        "allow_download": allow_download,
        "local_files_only": local_files_only,
        "download_success": success,
        "snapshot_path": snapshot_path,
        "error": error,
        "elapsed_time_sec": round(time.perf_counter() - start_time, 3),
        "disk_before": disk_before,
        "disk_after": _disk_summary(cache_path),
        "cache_files": local_files_summary or _cache_file_summary(cache_path),
    }
    write_download_report(summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = download_videomae_checkpoint(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        allow_download=args.allow_download,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(summary, indent=2))
    print(f"VIDEOMAE_DOWNLOAD_SUCCESS = {str(summary['download_success']).lower()}")
    if not summary["download_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
