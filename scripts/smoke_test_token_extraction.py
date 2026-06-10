"""End-to-end smoke test for the Step 3 token extraction pipeline."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.token_shards import load_token_shard, summarize_token_shard, validate_token_shard
from data.toy_data import generate_toy_video_dataset
from data.video_clip_dataset import VideoClipDataset
from scripts.extract_tokens import extract_tokens_from_config


REPORT_PATH = PROJECT_ROOT / "docs" / "TOKEN_EXTRACTION_SMOKE_REPORT.md"


def main() -> None:
    toy_dir = PROJECT_ROOT / "data" / "toy_videos"
    shard_dir = PROJECT_ROOT / "data" / "token_shards" / "dummy_toy"
    for path in (toy_dir, shard_dir):
        if path.exists():
            shutil.rmtree(path)

    records = generate_toy_video_dataset(
        output_dir=toy_dir,
        num_samples=16,
        total_frames=6,
        image_size=64,
        seed=42,
    )
    dataset = VideoClipDataset(root=toy_dir, metadata_file="metadata.jsonl", past_len=4, future_len=2)
    first = dataset[0]
    checks = {
        "past_shape_ok": list(first["past_video"].shape) == [4, 3, 64, 64],
        "future_shape_ok": list(first["future_video"].shape) == [2, 3, 64, 64],
    }
    summary = extract_tokens_from_config(
        config_path=PROJECT_ROOT / "configs" / "token_extraction_dummy.yaml",
        output_dir=shard_dir,
        device_name="cuda_if_available",
        overwrite=True,
    )
    shard_paths = sorted(shard_dir.glob("tokens_shard_*.pt"))
    checks["shard_count_ok"] = len(shard_paths) >= 2
    first_shard = load_token_shard(shard_paths[0], map_location="cpu")
    validate_token_shard(first_shard, strict=True)
    first_summary = summarize_token_shard(first_shard)
    checks["past_tokens_shape_ok"] = first_summary["past_tokens_shape"] == [8, 196, 768]
    checks["future_tokens_shape_ok"] = first_summary["future_tokens_shape"] == [8, 196, 768]
    checks["sample_ids_len_ok"] = first_summary["num_samples"] == 8
    pass_flag = all(checks.values())

    report = [
        "# Token Extraction Smoke Report",
        "",
        "Command: `python scripts/smoke_test_token_extraction.py`",
        "",
        f"- dataset path: `{toy_dir}`",
        f"- dataset size: `{len(records)}`",
        f"- output shard path: `{shard_dir}`",
        f"- generated shard count: `{len(shard_paths)}`",
        f"- generated shard files: `{[path.name for path in shard_paths]}`",
        f"- validation result: `{pass_flag}`",
        "",
        "## First Shard Summary",
        "",
        "```json",
        json.dumps(first_summary, indent=2),
        "```",
        "",
        "## Extraction Summary",
        "",
        "```json",
        json.dumps(
            {
                "dataset_size": summary["dataset_size"],
                "batch_size": summary["batch_size"],
                "shard_size": summary["shard_size"],
                "device": summary["device"],
                "encoder_name": summary["encoder_name"],
                "num_shards": summary["num_shards"],
            },
            indent=2,
        ),
        "```",
        "",
        f"TOKEN_EXTRACTION_SMOKE_PASS = {str(pass_flag).lower()}",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    if not pass_flag:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

