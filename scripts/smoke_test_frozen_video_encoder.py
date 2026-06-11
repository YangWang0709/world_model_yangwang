"""Smoke test Step 9B frozen encoder interface with graceful dummy fallback."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from encoders.frozen_video_encoder import build_frozen_video_encoder
from scripts.check_video_encoder_capabilities import collect_video_encoder_capabilities, write_capability_report
from scripts.extract_tokens import load_yaml


REPORT_PATH = PROJECT_ROOT / "docs" / "FROZEN_VIDEO_ENCODER_SMOKE_REPORT.md"


def _gpu_memory_gib() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"free_gib": None, "total_gib": None}
    free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device("cuda"))
    return {
        "free_gib": round(free_bytes / (1024**3), 3),
        "total_gib": round(total_bytes / (1024**3), 3),
    }


def run_frozen_encoder_smoke(config_path: str | Path = PROJECT_ROOT / "configs" / "frozen_video_encoder_smoke.yaml") -> dict[str, Any]:
    capabilities = collect_video_encoder_capabilities()
    write_capability_report(capabilities)
    config = load_yaml(config_path)
    encoder_cfg = dict(config["encoder"])
    fallback_cfg = dict(config.get("fallback", {}))
    resource_cfg = dict(config.get("resource_limits", {}))
    requested_encoder = str(encoder_cfg.get("name", "videomae"))
    encoder = build_frozen_video_encoder(encoder_cfg)
    used_fallback = False
    fallback_reason = None
    actual_encoder = encoder.encoder_name
    if not encoder.is_available():
        fallback_reason = str(encoder.availability.get("reason", "requested encoder unavailable"))
        if not fallback_cfg.get("allow_dummy_fallback", False):
            raise RuntimeError(f"{requested_encoder} unavailable and dummy fallback disabled: {fallback_reason}")
        used_fallback = True
        actual_encoder = "dummy_video_encoder"
        encoder = build_frozen_video_encoder(
            {
                "name": "dummy_video_encoder",
                "num_tokens": int(fallback_cfg.get("dummy_num_tokens", 196)),
                "token_dim": int(fallback_cfg.get("dummy_token_dim", 768)),
                "device": encoder_cfg.get("device", "cuda_if_available"),
            }
        )

    batch_size = int(resource_cfg.get("batch_size", 1))
    clip_len = int(resource_cfg.get("clip_len", 8))
    image_size = int(resource_cfg.get("image_size", 224))
    video_batch = torch.rand(batch_size, clip_len, 3, image_size, image_size)
    memory_before = _gpu_memory_gib()
    try:
        tokens = encoder.encode(video_batch)
    except RuntimeError as exc:
        fallback_reason = f"encode failed for {requested_encoder}: {exc}"
        if not fallback_cfg.get("allow_dummy_fallback", False):
            raise
        used_fallback = True
        actual_encoder = "dummy_video_encoder"
        encoder = build_frozen_video_encoder(
            {
                "name": "dummy_video_encoder",
                "num_tokens": int(fallback_cfg.get("dummy_num_tokens", 196)),
                "token_dim": int(fallback_cfg.get("dummy_token_dim", 768)),
                "device": encoder_cfg.get("device", "cuda_if_available"),
            }
        )
        tokens = encoder.encode(video_batch)
    memory_after = _gpu_memory_gib()

    checks = {
        "rank_ok": tokens.ndim == 3,
        "batch_ok": tokens.shape[0] == batch_size,
        "finite_ok": bool(torch.isfinite(tokens).all()),
    }
    return {
        "capabilities": capabilities,
        "requested_encoder": requested_encoder,
        "actual_encoder": actual_encoder,
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "output_shape": list(tokens.shape),
        "gpu_memory_before": memory_before,
        "gpu_memory_after": memory_after,
        "checks": checks,
        "pass": all(checks.values()),
    }


def write_frozen_smoke_report(
    encoder_summary: dict[str, Any],
    extraction_summary: dict[str, Any] | None = None,
) -> None:
    extraction_summary = extraction_summary or {
        "pass": False,
        "note": "real video frozen token extraction smoke has not been run yet",
    }
    report = [
        "# Frozen Video Encoder Smoke Report",
        "",
        "Commands:",
        "- `python scripts/smoke_test_frozen_video_encoder.py`",
        "- `python scripts/smoke_test_real_video_frozen_token_extraction.py`",
        "",
        "## Frozen Encoder Smoke",
        "",
        f"- requested encoder: `{encoder_summary['requested_encoder']}`",
        f"- actual encoder: `{encoder_summary['actual_encoder']}`",
        f"- used fallback: `{encoder_summary['used_fallback']}`",
        f"- fallback reason: `{encoder_summary['fallback_reason']}`",
        f"- output token shape: `{encoder_summary['output_shape']}`",
        f"- GPU memory before: `{encoder_summary['gpu_memory_before']}`",
        f"- GPU memory after: `{encoder_summary['gpu_memory_after']}`",
        "",
        "```json",
        json.dumps(encoder_summary["checks"], indent=2),
        "```",
        "",
        f"FROZEN_VIDEO_ENCODER_SMOKE_PASS = {str(encoder_summary['pass']).lower()}",
        "",
        "## Real Video Frozen Token Extraction Smoke",
        "",
        "```json",
        json.dumps(extraction_summary, indent=2),
        "```",
        "",
        f"REAL_VIDEO_FROZEN_TOKEN_EXTRACTION_SMOKE_PASS = {str(extraction_summary.get('pass', False)).lower()}",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    summary = run_frozen_encoder_smoke()
    write_frozen_smoke_report(summary)
    print(json.dumps(summary, indent=2))
    print(f"FROZEN_VIDEO_ENCODER_SMOKE_PASS = {str(summary['pass']).lower()}")
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
