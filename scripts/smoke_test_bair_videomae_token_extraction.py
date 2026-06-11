"""Smoke test BAIR small-subset frozen VideoMAE token extraction."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.token_shards import load_token_shard, summarize_token_shard
from scripts.extract_tokens import extract_tokens_from_config, load_yaml


CONFIG_PATH = PROJECT_ROOT / "configs" / "token_extraction_bair_videomae_smoke.yaml"
REPORT_PATH = PROJECT_ROOT / "docs" / "BAIR_VIDEOMAE_TOKEN_EXTRACTION_SMOKE_REPORT.md"


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _memory_snapshot() -> dict[str, float]:
    values: dict[str, float] = {}
    meminfo: dict[str, int] = {}
    try:
        with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
            for line in handle:
                key, raw_value = line.split(":", 1)
                meminfo[key] = int(raw_value.strip().split()[0])
    except Exception:
        return values
    total_kib = meminfo.get("MemTotal", 0)
    available_kib = meminfo.get("MemAvailable", 0)
    values["ram_total_gib"] = round(total_kib / 1024 / 1024, 3)
    values["ram_available_gib"] = round(available_kib / 1024 / 1024, 3)
    values["ram_used_gib"] = round((total_kib - available_kib) / 1024 / 1024, 3)
    return values


def _gpu_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_allocated_mib": 0.0,
        "torch_cuda_reserved_mib": 0.0,
        "nvidia_smi": None,
    }
    if torch.cuda.is_available():
        snapshot["torch_cuda_allocated_mib"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 3)
        snapshot["torch_cuda_reserved_mib"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 3)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        snapshot["nvidia_smi"] = {"error": str(exc)}
        return snapshot
    if result.returncode != 0:
        snapshot["nvidia_smi"] = {"error": result.stderr.strip()}
        return snapshot
    rows: list[dict[str, Any]] = []
    for row in result.stdout.splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) >= 4:
            rows.append(
                {
                    "name": parts[0],
                    "memory_total_mib": int(parts[1]),
                    "memory_used_mib": int(parts[2]),
                    "memory_free_mib": int(parts[3]),
                }
            )
    snapshot["nvidia_smi"] = rows
    return snapshot


def _resource_snapshot() -> dict[str, Any]:
    usage = shutil.disk_usage(PROJECT_ROOT)
    return {
        **_memory_snapshot(),
        "disk_free_gib": round(usage.free / 1024 / 1024 / 1024, 3),
        "gpu": _gpu_snapshot(),
    }


def _safe_clear_output(output_root: Path) -> None:
    expected_root = (PROJECT_ROOT / "data" / "token_shards" / "bair_videomae_smoke").resolve()
    resolved = output_root.resolve()
    if resolved != expected_root:
        raise ValueError(f"Refusing to delete unexpected token output path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _inspect_first_shard(path: Path, expected_split: str) -> dict[str, Any]:
    shard = load_token_shard(path, map_location="cpu")
    if shard["split"] != expected_split:
        raise AssertionError(f"Expected split {expected_split!r}, got {shard['split']!r}")
    if shard["encoder_name"] != "videomae":
        raise AssertionError(f"Expected videomae encoder, got {shard['encoder_name']!r}")
    if bool(shard["encoder_config"].get("used_fallback")):
        raise AssertionError("VideoMAE extraction used fallback, but fallback must be disabled")
    if not torch.isfinite(shard["past_tokens"]).all():
        raise AssertionError("past_tokens contain non-finite values")
    if not torch.isfinite(shard["future_tokens"]).all():
        raise AssertionError("future_tokens contain non-finite values")

    summary = summarize_token_shard(shard)
    summary["path"] = str(path)
    summary["first_sample_id"] = shard["sample_ids"][0]
    summary["first_metadata"] = shard["metadata"][0]
    summary["encoder_config"] = {
        "requested_encoder": shard["encoder_config"].get("requested_encoder"),
        "actual_encoder": shard["encoder_config"].get("actual_encoder"),
        "used_fallback": shard["encoder_config"].get("used_fallback"),
        "model_name_or_path": shard["encoder_config"].get("model_name_or_path"),
        "output_mode": shard["encoder_config"].get("output_mode"),
        "num_tokens": shard["encoder_config"].get("num_tokens"),
        "token_dim": shard["encoder_config"].get("token_dim"),
        "temporal_pad_to": shard["encoder_config"].get("temporal_pad_to"),
    }
    return summary


def _format_resource(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, indent=2, sort_keys=True)


def _write_report(result: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BAIR VideoMAE Token Extraction Smoke Report",
        "",
        f"- command: `{result['command']}`",
        f"- input subset dir: `{result['subset_root']}`",
        f"- VideoMAE model path: `{result['model_path']}`",
        f"- train samples: `{result['train_samples']}`",
        f"- test samples: `{result['test_samples']}`",
        f"- train output dir: `{result['train_output_dir']}`",
        f"- test output dir: `{result['test_output_dir']}`",
        f"- train shard count: `{result['train_shard_count']}`",
        f"- test shard count: `{result['test_shard_count']}`",
        f"- first train shard shape: `{result['first_train_shard']['past_tokens_shape']}`",
        f"- first test shard shape: `{result['first_test_shard']['past_tokens_shape']}`",
        f"- elapsed_time_sec: `{result['elapsed_time_sec']}`",
        f"- oom: `{result['oom']}`",
        f"- cloud_recommendation: `{result['cloud_recommendation']}`",
        f"- BAIR_VIDEOMAE_TOKEN_EXTRACTION_SMOKE_PASS: `{str(result['smoke_pass']).lower()}`",
        "",
        "## Resource Before",
        "",
        "```json",
        _format_resource(result["resource_before"]),
        "```",
        "",
        "## Resource After",
        "",
        "```json",
        _format_resource(result["resource_after"]),
        "```",
        "",
        "## First Train Shard",
        "",
        "```json",
        json.dumps(result["first_train_shard"], indent=2, sort_keys=True),
        "```",
        "",
        "## First Test Shard",
        "",
        "```json",
        json.dumps(result["first_test_shard"], indent=2, sort_keys=True),
        "```",
        "",
        "## Extraction Summary",
        "",
        "```json",
        json.dumps(result["extraction_summary"], indent=2, sort_keys=True),
        "```",
        "",
        "BAIR_VIDEOMAE_TOKEN_EXTRACTION_SMOKE_PASS = true",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    config = load_yaml(CONFIG_PATH)
    subset_root = Path(config["dataset"]["root"])
    train_metadata = Path(config["dataset"]["train_metadata_file"])
    test_metadata = Path(config["dataset"]["test_metadata_file"])
    model_path = Path(config["encoder"]["model_name_or_path"])
    output_root = Path(config["extraction"]["output_root"])

    for required_path in (subset_root, train_metadata, test_metadata, model_path):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required Step 11B input is missing: {required_path}. "
                "Run Step 11A-fix first; this smoke test will not download BAIR or models."
            )

    _safe_clear_output(output_root)
    resource_before = _resource_snapshot()
    oom = False
    try:
        extraction_summary = extract_tokens_from_config(CONFIG_PATH, overwrite=True)
    except RuntimeError as exc:
        message = str(exc).lower()
        oom = "out of memory" in message or "oom" in message
        if oom:
            raise RuntimeError(
                "BAIR VideoMAE token extraction hit OOM with batch_size=1. "
                "Stop this stage and use a cloud 4090 or 48GB GPU server before retrying."
            ) from exc
        raise
    resource_after = _resource_snapshot()

    train_summary = extraction_summary["split_summaries"]["train"]
    test_summary = extraction_summary["split_summaries"]["test"]
    if train_summary["num_shards"] <= 0 or test_summary["num_shards"] <= 0:
        raise AssertionError("Expected train and test token shards to be generated")

    first_train_path = Path(train_summary["shards"][0]["path"])
    first_test_path = Path(test_summary["shards"][0]["path"])
    first_train = _inspect_first_shard(first_train_path, expected_split="train")
    first_test = _inspect_first_shard(first_test_path, expected_split="test")

    result = {
        "command": (
            "/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab "
            "python scripts/smoke_test_bair_videomae_token_extraction.py"
        ),
        "subset_root": str(subset_root),
        "model_path": str(model_path),
        "train_samples": _count_jsonl(train_metadata),
        "test_samples": _count_jsonl(test_metadata),
        "train_output_dir": train_summary["output_dir"],
        "test_output_dir": test_summary["output_dir"],
        "train_shard_count": train_summary["num_shards"],
        "test_shard_count": test_summary["num_shards"],
        "first_train_shard": first_train,
        "first_test_shard": first_test,
        "resource_before": resource_before,
        "resource_after": resource_after,
        "elapsed_time_sec": round(time.perf_counter() - start, 3),
        "oom": oom,
        "cloud_recommendation": "not_needed_for_step11b",
        "smoke_pass": True,
        "extraction_summary": extraction_summary,
    }
    _write_report(result)
    print(REPORT_PATH.read_text(encoding="utf-8"))
    print("BAIR_VIDEOMAE_TOKEN_EXTRACTION_SMOKE_PASS = true")


if __name__ == "__main__":
    main()
