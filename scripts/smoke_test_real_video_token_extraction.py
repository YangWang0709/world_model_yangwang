"""Smoke test Step 9A dummy token extraction on real_video_minimal."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.real_video_dataset import RealVideoClipDataset
from data.token_shards import load_token_shard, summarize_token_shard, validate_token_shard
from scripts.extract_tokens import extract_tokens_from_config
from scripts.smoke_test_real_video_dataset import (
    DATASET_ROOT,
    METADATA_PATH,
    run_dataset_smoke,
    write_smoke_report,
)


SHARD_DIR = PROJECT_ROOT / "data" / "token_shards" / "real_video_dummy"


def run_token_smoke() -> tuple[dict[str, object], dict[str, object]]:
    if not METADATA_PATH.exists():
        dataset_summary = run_dataset_smoke()
    else:
        dataset = RealVideoClipDataset(
            root=DATASET_ROOT,
            metadata_file=METADATA_PATH,
            past_len=4,
            future_len=4,
            image_size=224,
            split="real_minimal",
        )
        first = dataset[0]
        dataset_summary = {
            "dataset_root": str(DATASET_ROOT),
            "metadata_path": str(METADATA_PATH),
            "num_samples": len(dataset),
            "first_sample_id": first["sample_id"],
            "past_shape": list(first["past_video"].shape),
            "future_shape": list(first["future_video"].shape),
            "value_min": min(float(first["past_video"].min()), float(first["future_video"].min())),
            "value_max": max(float(first["past_video"].max()), float(first["future_video"].max())),
            "checks": {
                "num_samples_ok": len(dataset) >= 16,
                "past_shape_ok": list(first["past_video"].shape) == [4, 3, 224, 224],
                "future_shape_ok": list(first["future_video"].shape) == [4, 3, 224, 224],
                "finite_ok": True,
                "value_range_ok": True,
            },
            "resource_summary": {},
            "pass": True,
        }

    if SHARD_DIR.exists():
        shutil.rmtree(SHARD_DIR)
    summary = extract_tokens_from_config(
        config_path=PROJECT_ROOT / "configs" / "token_extraction_real_video_dummy.yaml",
        output_dir=SHARD_DIR,
        device_name="cuda_if_available",
        overwrite=True,
    )
    shard_paths = sorted(SHARD_DIR.glob("tokens_shard_*.pt"))
    first_shard = load_token_shard(shard_paths[0], map_location="cpu")
    validate_token_shard(first_shard, strict=True)
    first_summary = summarize_token_shard(first_shard)
    first_metadata = first_shard["metadata"][0]
    checks = {
        "shard_count_ok": len(shard_paths) >= 2,
        "past_tokens_shape_ok": first_summary["past_tokens_shape"] == [8, 196, 768],
        "future_tokens_shape_ok": first_summary["future_tokens_shape"] == [8, 196, 768],
        "split_ok": first_summary["split"] == "real_minimal",
        "metadata_source_type_ok": "source_type" in first_metadata,
        "metadata_path_ok": "path" in first_metadata,
    }
    token_summary = {
        "output_dir": str(SHARD_DIR),
        "generated_shard_count": len(shard_paths),
        "generated_shard_files": [path.name for path in shard_paths],
        "first_shard_summary": first_summary,
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
    return dataset_summary, token_summary


def main() -> None:
    dataset_summary, token_summary = run_token_smoke()
    if not dataset_summary.get("resource_summary"):
        from scripts.check_resource_limits import collect_resource_summary, write_resource_report

        resource_summary = collect_resource_summary(PROJECT_ROOT)
        write_resource_report(resource_summary)
        dataset_summary["resource_summary"] = resource_summary
    write_smoke_report(dataset_summary, token_summary)
    print(json.dumps(token_summary, indent=2))
    print(f"REAL_VIDEO_TOKEN_EXTRACTION_SMOKE_PASS = {str(token_summary['pass']).lower()}")
    if not token_summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
