"""Smoke test the Step 9A real_video_minimal dataset path."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.real_video_dataset import RealVideoClipDataset
from data.real_video_index import build_real_video_index
from scripts.check_resource_limits import collect_resource_summary, write_resource_report
from scripts.create_real_video_minimal_subset import create_real_video_minimal_subset


DATASET_ROOT = PROJECT_ROOT / "data" / "real_video_minimal"
RAW_PT_DIR = DATASET_ROOT / "raw" / "pt_clips"
METADATA_PATH = DATASET_ROOT / "metadata.jsonl"
REPORT_PATH = PROJECT_ROOT / "docs" / "REAL_VIDEO_SMOKE_REPORT.md"


def run_dataset_smoke() -> dict[str, object]:
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)

    subset_summary = create_real_video_minimal_subset(
        output_dir=RAW_PT_DIR,
        num_samples=16,
        total_frames=8,
        image_size=128,
        seed=42,
    )
    index_summary = build_real_video_index(
        input_dir=DATASET_ROOT / "raw",
        output_metadata=METADATA_PATH,
        source="real_video_minimal",
        max_samples=100,
        min_frames=8,
        task_text="predict future visual dynamics",
        split="real_minimal",
    )
    dataset = RealVideoClipDataset(
        root=DATASET_ROOT,
        metadata_file=METADATA_PATH,
        past_len=4,
        future_len=4,
        image_size=224,
        split="real_minimal",
    )
    first = dataset[0]
    past_video = first["past_video"]
    future_video = first["future_video"]
    value_min = min(float(past_video.min()), float(future_video.min()))
    value_max = max(float(past_video.max()), float(future_video.max()))
    checks = {
        "num_samples_ok": len(dataset) == 16,
        "past_shape_ok": list(past_video.shape) == [4, 3, 224, 224],
        "future_shape_ok": list(future_video.shape) == [4, 3, 224, 224],
        "finite_ok": bool(torch.isfinite(past_video).all() and torch.isfinite(future_video).all()),
        "value_range_ok": value_min >= -1e-6 and value_max <= 1.0 + 1e-6,
    }
    pass_flag = all(checks.values())
    resource_summary = collect_resource_summary(PROJECT_ROOT)
    write_resource_report(resource_summary)
    return {
        "dataset_root": str(DATASET_ROOT),
        "raw_pt_dir": str(RAW_PT_DIR),
        "metadata_path": str(METADATA_PATH),
        "subset_summary": {key: value for key, value in subset_summary.items() if key != "clip_paths"},
        "index_summary": index_summary,
        "num_samples": len(dataset),
        "first_sample_id": first["sample_id"],
        "past_shape": list(past_video.shape),
        "future_shape": list(future_video.shape),
        "value_min": value_min,
        "value_max": value_max,
        "checks": checks,
        "resource_summary": resource_summary,
        "pass": pass_flag,
    }


def write_smoke_report(dataset_summary: dict[str, object], token_summary: dict[str, object] | None = None) -> None:
    token_summary = token_summary or {
        "pass": False,
        "note": "token extraction smoke has not been run yet",
    }
    report = [
        "# Real Video Smoke Report",
        "",
        "Commands:",
        "- `python scripts/smoke_test_real_video_dataset.py`",
        "- `python scripts/smoke_test_real_video_token_extraction.py`",
        "",
        "## Resource Summary",
        "",
        "```json",
        json.dumps(dataset_summary["resource_summary"], indent=2),
        "```",
        "",
        "## Dataset Smoke",
        "",
        f"- dataset path: `{dataset_summary['dataset_root']}`",
        f"- metadata path: `{dataset_summary['metadata_path']}`",
        f"- number of samples: `{dataset_summary['num_samples']}`",
        f"- first sample id: `{dataset_summary['first_sample_id']}`",
        f"- first past shape: `{dataset_summary['past_shape']}`",
        f"- first future shape: `{dataset_summary['future_shape']}`",
        f"- value range: `[{dataset_summary['value_min']}, {dataset_summary['value_max']}]`",
        "",
        "```json",
        json.dumps(dataset_summary["checks"], indent=2),
        "```",
        "",
        f"REAL_VIDEO_DATASET_SMOKE_PASS = {str(dataset_summary['pass']).lower()}",
        "",
        "## Token Extraction Smoke",
        "",
        "```json",
        json.dumps(token_summary, indent=2),
        "```",
        "",
        f"REAL_VIDEO_TOKEN_EXTRACTION_SMOKE_PASS = {str(token_summary.get('pass', False)).lower()}",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    dataset_summary = run_dataset_smoke()
    write_smoke_report(dataset_summary)
    print(json.dumps(dataset_summary, indent=2))
    print(f"REAL_VIDEO_DATASET_SMOKE_PASS = {str(dataset_summary['pass']).lower()}")
    if not dataset_summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
