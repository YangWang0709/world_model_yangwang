"""Smoke test a real frozen VideoMAE encoder under Step 9C limits."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from encoders.frozen_video_encoder import build_frozen_video_encoder
from scripts.check_video_encoder_capabilities import collect_video_encoder_capabilities
from scripts.download_videomae_checkpoint import download_videomae_checkpoint
from scripts.extract_tokens import load_yaml


CONFIG_PATH = PROJECT_ROOT / "configs" / "videomae_real_video_smoke.yaml"
REPORT_PATH = PROJECT_ROOT / "docs" / "VIDEOMAE_REAL_ENCODER_SMOKE_REPORT.md"


def _bytes_to_gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024**3), 3)


def _ram_summary() -> dict[str, float | None]:
    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        return {
            "total_gib": _bytes_to_gib(memory.total),
            "available_gib": _bytes_to_gib(memory.available),
            "used_gib": _bytes_to_gib(memory.used),
        }
    except ImportError:
        meminfo: dict[str, int] = {}
        meminfo_path = Path("/proc/meminfo")
        if meminfo_path.exists():
            for line in meminfo_path.read_text(encoding="utf-8").splitlines():
                key, value = line.split(":", 1)
                meminfo[key] = int(value.strip().split()[0]) * 1024
        total = meminfo.get("MemTotal")
        available = meminfo.get("MemAvailable")
        used = total - available if total is not None and available is not None else None
        return {
            "total_gib": _bytes_to_gib(total),
            "available_gib": _bytes_to_gib(available),
            "used_gib": _bytes_to_gib(used),
        }


def _gpu_memory_gib() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"free_gib": None, "total_gib": None, "used_gib": None}
    free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device("cuda"))
    return {
        "free_gib": _bytes_to_gib(free_bytes),
        "total_gib": _bytes_to_gib(total_bytes),
        "used_gib": _bytes_to_gib(total_bytes - free_bytes),
    }


def _is_oom_error(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _cloud_recommendation(pass_value: bool, oom: bool, ram_after: dict[str, Any]) -> str:
    used_gib = ram_after.get("used_gib")
    if oom or (used_gib is not None and used_gib > 28):
        return "recommended: retry Step 9C/10 on a cloud 4090 or 48GB GPU server"
    if pass_value:
        return "not required for Step 9C; local RTX 5080 conservative smoke succeeded"
    return "not required yet; inspect failure before changing server"


def write_videomae_smoke_report(
    encoder_summary: dict[str, Any],
    extraction_summary: dict[str, Any] | None = None,
    report_path: str | Path = REPORT_PATH,
) -> Path:
    extraction_summary = extraction_summary or {
        "pass": False,
        "note": "real_video VideoMAE token extraction smoke has not been run yet",
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VideoMAE Real Encoder Smoke Report",
        "",
        "Commands:",
        "- `python scripts/smoke_test_videomae_real_encoder.py`",
        "- `python scripts/smoke_test_real_video_videomae_token_extraction.py`",
        "",
        "## Encoder Smoke",
        "",
        f"- model_name: `{encoder_summary.get('model_name')}`",
        f"- cache_dir: `{encoder_summary.get('cache_dir')}`",
        f"- device: `{encoder_summary.get('device')}`",
        f"- input_shape: `{encoder_summary.get('input_shape')}`",
        f"- output_shape: `{encoder_summary.get('output_shape')}`",
        f"- dtype: `{encoder_summary.get('dtype')}`",
        f"- GPU memory before: `{encoder_summary.get('gpu_memory_before')}`",
        f"- GPU memory after: `{encoder_summary.get('gpu_memory_after')}`",
        f"- RAM before: `{encoder_summary.get('ram_before')}`",
        f"- RAM after: `{encoder_summary.get('ram_after')}`",
        f"- elapsed_time_sec: `{encoder_summary.get('elapsed_time_sec')}`",
        f"- oom: `{encoder_summary.get('oom')}`",
        f"- cloud_recommendation: `{encoder_summary.get('cloud_recommendation')}`",
        f"- pass: `{encoder_summary.get('pass')}`",
        "",
        "```json",
        json.dumps(encoder_summary.get("checks", {}), indent=2),
        "```",
        "",
        f"VIDEOMAE_REAL_ENCODER_SMOKE_PASS = {str(encoder_summary.get('pass', False)).lower()}",
        "",
        "## Real Video Token Extraction",
        "",
        "```json",
        json.dumps(extraction_summary, indent=2),
        "```",
        "",
        "REAL_VIDEO_VIDEOMAE_TOKEN_EXTRACTION_SMOKE_PASS = "
        f"{str(extraction_summary.get('pass', False)).lower()}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_videomae_real_encoder_smoke(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    resource_cfg = dict(config.get("resource_limits", {}))
    encoder_cfg = dict(config["encoder"])
    capabilities = collect_video_encoder_capabilities()
    batch_size = int(resource_cfg.get("batch_size", 1))
    clip_len = int(resource_cfg.get("clip_len", encoder_cfg.get("num_frames", 8)))
    image_size = int(resource_cfg.get("image_size", encoder_cfg.get("image_size", 224)))

    if batch_size != 1:
        raise ValueError("Step 9C VideoMAE smoke requires batch_size=1")
    if int(resource_cfg.get("num_workers", 0)) != 0:
        raise ValueError("Step 9C VideoMAE smoke requires num_workers=0")
    if clip_len > 8 or image_size > 224:
        raise ValueError("Step 9C VideoMAE smoke requires clip_len<=8 and image_size<=224")

    download_summary = download_videomae_checkpoint(
        model_name=str(encoder_cfg["model_name_or_path"]),
        cache_dir=encoder_cfg["cache_dir"],
        allow_download=bool(encoder_cfg.get("allow_download", False)),
        local_files_only=bool(encoder_cfg.get("local_files_only", False)),
    )
    if not download_summary["download_success"]:
        summary = {
            "model_name": encoder_cfg.get("model_name_or_path"),
            "cache_dir": encoder_cfg.get("cache_dir"),
            "device": encoder_cfg.get("device"),
            "input_shape": [batch_size, clip_len, 3, image_size, image_size],
            "output_shape": None,
            "dtype": None,
            "gpu_memory_before": _gpu_memory_gib(),
            "gpu_memory_after": _gpu_memory_gib(),
            "ram_before": _ram_summary(),
            "ram_after": _ram_summary(),
            "elapsed_time_sec": 0.0,
            "oom": False,
            "download_summary": download_summary,
            "capabilities": capabilities,
            "cloud_recommendation": "not required yet; provide the checkpoint from Windows local first",
            "checks": {"download_ok": False},
            "pass": False,
            "error": download_summary.get("error"),
        }
        write_videomae_smoke_report(summary)
        return summary

    encoder = build_frozen_video_encoder(encoder_cfg)
    if not encoder.is_available():
        summary = {
            "model_name": encoder_cfg.get("model_name_or_path"),
            "cache_dir": encoder_cfg.get("cache_dir"),
            "device": encoder_cfg.get("device"),
            "input_shape": [batch_size, clip_len, 3, image_size, image_size],
            "output_shape": None,
            "dtype": None,
            "gpu_memory_before": _gpu_memory_gib(),
            "gpu_memory_after": _gpu_memory_gib(),
            "ram_before": _ram_summary(),
            "ram_after": _ram_summary(),
            "elapsed_time_sec": 0.0,
            "oom": False,
            "download_summary": download_summary,
            "capabilities": capabilities,
            "encoder_availability": encoder.availability,
            "cloud_recommendation": "not required yet; model did not load",
            "checks": {"download_ok": True, "encoder_available": False},
            "pass": False,
            "error": encoder.availability.get("reason"),
        }
        write_videomae_smoke_report(summary)
        return summary

    torch.manual_seed(int(config.get("seed", 42)))
    video_batch = torch.rand(batch_size, clip_len, 3, image_size, image_size)
    ram_before = _ram_summary()
    gpu_before = _gpu_memory_gib()
    start_time = time.perf_counter()
    oom = False
    error: str | None = None
    tokens: torch.Tensor | None = None
    try:
        tokens = encoder.encode(video_batch)
    except Exception as exc:  # pragma: no cover - hardware/network dependent
        oom = _is_oom_error(exc)
        error = str(exc)
    elapsed = round(time.perf_counter() - start_time, 3)
    gpu_after = _gpu_memory_gib()
    ram_after = _ram_summary()

    checks = {
        "download_ok": bool(download_summary["download_success"]),
        "encoder_available": encoder.is_available(),
        "rank_ok": bool(tokens is not None and tokens.ndim == 3),
        "batch_ok": bool(tokens is not None and tokens.shape[0] == batch_size),
        "finite_ok": bool(tokens is not None and torch.isfinite(tokens).all()),
        "nonempty_tokens": bool(tokens is not None and tokens.shape[1] > 0 and tokens.shape[2] > 0),
        "oom_ok": not oom,
    }
    pass_value = all(checks.values())
    summary = {
        "model_name": encoder_cfg.get("model_name_or_path"),
        "cache_dir": encoder_cfg.get("cache_dir"),
        "device": encoder.availability.get("device", encoder_cfg.get("device")),
        "input_shape": [batch_size, clip_len, 3, image_size, image_size],
        "output_shape": list(tokens.shape) if tokens is not None else None,
        "dtype": str(tokens.dtype) if tokens is not None else None,
        "gpu_memory_before": gpu_before,
        "gpu_memory_after": gpu_after,
        "ram_before": ram_before,
        "ram_after": ram_after,
        "elapsed_time_sec": elapsed,
        "oom": oom,
        "download_summary": download_summary,
        "capabilities": capabilities,
        "encoder_availability": encoder.availability,
        "encoder_last_encode_summary": getattr(encoder.impl, "last_encode_summary", {}),
        "cloud_recommendation": _cloud_recommendation(pass_value, oom, ram_after),
        "checks": checks,
        "pass": pass_value,
        "error": error,
    }
    write_videomae_smoke_report(summary)
    return summary


def main() -> None:
    summary = run_videomae_real_encoder_smoke()
    print(json.dumps(summary, indent=2))
    print(f"VIDEOMAE_REAL_ENCODER_SMOKE_PASS = {str(summary['pass']).lower()}")
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
