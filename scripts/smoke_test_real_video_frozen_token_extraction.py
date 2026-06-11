"""Smoke test real_video_minimal token extraction through the frozen encoder factory."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.token_shards import load_token_shard, summarize_token_shard, validate_token_shard
from scripts.extract_tokens import extract_tokens_from_config
from scripts.smoke_test_frozen_video_encoder import run_frozen_encoder_smoke, write_frozen_smoke_report
from scripts.smoke_test_real_video_dataset import METADATA_PATH, run_dataset_smoke


SHARD_DIR = PROJECT_ROOT / "data" / "token_shards" / "real_video_frozen_encoder_smoke"


def run_real_video_frozen_token_smoke() -> tuple[dict[str, Any], dict[str, Any]]:
    if not METADATA_PATH.exists():
        run_dataset_smoke()
    encoder_summary = run_frozen_encoder_smoke()
    if SHARD_DIR.exists():
        shutil.rmtree(SHARD_DIR)

    summary = extract_tokens_from_config(
        config_path=PROJECT_ROOT / "configs" / "token_extraction_real_video_frozen_encoder.yaml",
        output_dir=SHARD_DIR,
        device_name="cuda_if_available",
        overwrite=True,
    )
    shard_paths = sorted(SHARD_DIR.glob("tokens_shard_*.pt"))
    first_shard = load_token_shard(shard_paths[0], map_location="cpu")
    validate_token_shard(first_shard, strict=True)
    first_summary = summarize_token_shard(first_shard)
    checks = {
        "shard_exists": bool(shard_paths),
        "past_tokens_finite": bool(torch.isfinite(first_shard["past_tokens"]).all()),
        "future_tokens_finite": bool(torch.isfinite(first_shard["future_tokens"]).all()),
        "split_ok": first_summary["split"] == "real_minimal",
        "encoder_name_ok": first_shard["encoder_name"] in {"dummy_video_encoder", "videomae", "vjepa"},
        "summary_exists": (SHARD_DIR / "extraction_summary.json").exists(),
    }
    extraction_summary = {
        "output_dir": str(SHARD_DIR),
        "generated_shard_count": len(shard_paths),
        "generated_shard_files": [path.name for path in shard_paths],
        "first_shard_summary": first_summary,
        "requested_encoder": summary.get("requested_encoder"),
        "actual_encoder": summary.get("actual_encoder"),
        "used_fallback": summary.get("used_fallback"),
        "fallback_reason": summary.get("fallback_reason"),
        "extraction_summary": {
            "dataset_size": summary["dataset_size"],
            "batch_size": summary["batch_size"],
            "shard_size": summary["shard_size"],
            "device": summary["device"],
            "encoder_name": summary["encoder_name"],
            "num_shards": summary["num_shards"],
        },
        "checks": checks,
        "pass": all(checks.values()),
    }
    return encoder_summary, extraction_summary


def main() -> None:
    encoder_summary, extraction_summary = run_real_video_frozen_token_smoke()
    write_frozen_smoke_report(encoder_summary, extraction_summary)
    print(json.dumps(extraction_summary, indent=2))
    print(
        "REAL_VIDEO_FROZEN_TOKEN_EXTRACTION_SMOKE_PASS = "
        f"{str(extraction_summary['pass']).lower()}"
    )
    if not extraction_summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
