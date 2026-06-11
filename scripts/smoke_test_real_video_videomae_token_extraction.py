"""Smoke test real_video_minimal token extraction with the real VideoMAE wrapper."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.token_shards import load_token_shard, summarize_token_shard, validate_token_shard
from scripts.extract_tokens import extract_tokens_from_config
from scripts.smoke_test_real_video_dataset import METADATA_PATH, run_dataset_smoke
from scripts.smoke_test_videomae_real_encoder import (
    _gpu_memory_gib,
    _is_oom_error,
    _ram_summary,
    run_videomae_real_encoder_smoke,
    write_videomae_smoke_report,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "token_extraction_real_video_videomae.yaml"
SHARD_DIR = PROJECT_ROOT / "data" / "token_shards" / "real_video_videomae_smoke"


def run_real_video_videomae_token_smoke() -> tuple[dict[str, Any], dict[str, Any]]:
    if not METADATA_PATH.exists():
        run_dataset_smoke()

    encoder_summary = run_videomae_real_encoder_smoke()
    if not encoder_summary["pass"]:
        extraction_summary = {
            "output_dir": str(SHARD_DIR),
            "generated_shard_count": 0,
            "checks": {"encoder_smoke_passed": False},
            "pass": False,
            "error": encoder_summary.get("error"),
        }
        write_videomae_smoke_report(encoder_summary, extraction_summary)
        return encoder_summary, extraction_summary

    if SHARD_DIR.exists():
        shutil.rmtree(SHARD_DIR)

    ram_before = _ram_summary()
    gpu_before = _gpu_memory_gib()
    start_time = time.perf_counter()
    oom = False
    error: str | None = None
    summary: dict[str, Any] | None = None
    try:
        summary = extract_tokens_from_config(
            config_path=CONFIG_PATH,
            output_dir=SHARD_DIR,
            device_name="cuda_if_available",
            overwrite=True,
        )
    except Exception as exc:  # pragma: no cover - hardware dependent
        oom = _is_oom_error(exc)
        error = str(exc)

    elapsed = round(time.perf_counter() - start_time, 3)
    shard_paths = sorted(SHARD_DIR.glob("tokens_shard_*.pt")) if SHARD_DIR.exists() else []
    first_summary: dict[str, Any] | None = None
    first_encoder_config: dict[str, Any] | None = None
    finite_past = False
    finite_future = False
    split_ok = False
    encoder_name_ok = False
    if shard_paths:
        first_shard = load_token_shard(shard_paths[0], map_location="cpu")
        validate_token_shard(first_shard, strict=True)
        first_summary = summarize_token_shard(first_shard)
        first_encoder_config = dict(first_shard.get("encoder_config", {}))
        finite_past = bool(torch.isfinite(first_shard["past_tokens"]).all())
        finite_future = bool(torch.isfinite(first_shard["future_tokens"]).all())
        split_ok = first_summary["split"] == "real_minimal"
        encoder_name_ok = first_shard["encoder_name"] == "videomae"

    summary_path = SHARD_DIR / "extraction_summary.json"
    checks = {
        "encoder_smoke_passed": encoder_summary["pass"],
        "shard_exists": bool(shard_paths),
        "past_tokens_finite": finite_past,
        "future_tokens_finite": finite_future,
        "split_ok": split_ok,
        "encoder_name_ok": encoder_name_ok,
        "summary_exists": summary_path.exists(),
        "requested_encoder_ok": bool(summary and summary.get("requested_encoder") == "videomae"),
        "actual_encoder_ok": bool(summary and summary.get("actual_encoder") == "videomae"),
        "used_fallback_false": bool(summary and summary.get("used_fallback") is False),
        "oom_ok": not oom,
    }
    extraction_summary = {
        "output_dir": str(SHARD_DIR),
        "generated_shard_count": len(shard_paths),
        "generated_shard_files": [path.name for path in shard_paths],
        "first_shard_summary": first_summary,
        "first_shard_encoder_config": first_encoder_config,
        "requested_encoder": summary.get("requested_encoder") if summary else None,
        "actual_encoder": summary.get("actual_encoder") if summary else None,
        "used_fallback": summary.get("used_fallback") if summary else None,
        "model_name_or_path": summary.get("model_name_or_path") if summary else None,
        "cache_dir": summary.get("cache_dir") if summary else None,
        "extraction_summary_path": str(summary_path),
        "extraction_summary": {
            "dataset_size": summary.get("dataset_size") if summary else None,
            "batch_size": summary.get("batch_size") if summary else None,
            "max_samples": summary.get("max_samples") if summary else None,
            "shard_size": summary.get("shard_size") if summary else None,
            "device": summary.get("device") if summary else None,
            "encoder_name": summary.get("encoder_name") if summary else None,
            "num_shards": summary.get("num_shards") if summary else None,
            "elapsed_time_sec": summary.get("elapsed_time_sec") if summary else None,
        },
        "resource": {
            "ram_before": ram_before,
            "ram_after": _ram_summary(),
            "gpu_memory_before": gpu_before,
            "gpu_memory_after": _gpu_memory_gib(),
            "elapsed_time_sec": elapsed,
            "oom": oom,
        },
        "checks": checks,
        "pass": all(checks.values()),
        "error": error,
    }
    write_videomae_smoke_report(encoder_summary, extraction_summary)
    return encoder_summary, extraction_summary


def main() -> None:
    _, extraction_summary = run_real_video_videomae_token_smoke()
    print(json.dumps(extraction_summary, indent=2))
    print(
        "REAL_VIDEO_VIDEOMAE_TOKEN_EXTRACTION_SMOKE_PASS = "
        f"{str(extraction_summary['pass']).lower()}"
    )
    if not extraction_summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
