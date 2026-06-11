"""Smoke test predictive importance on BAIR VideoMAE token shards."""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.importance_shards import load_importance_shard, summarize_importance_shard, validate_importance_shard
from data.token_shards import load_token_shard
from eval.eval_importance_bair_videomae_summary import evaluate_bair_importance_root
from scripts.generate_predictive_importance import generate_predictive_importance, load_yaml
from scripts.inspect_importance_shard import inspect_importance_shard


CONFIG_PATH = PROJECT_ROOT / "configs" / "generate_importance_bair_videomae_smoke.yaml"
REPORT_PATH = PROJECT_ROOT / "docs" / "IMPORTANCE_BAIR_VIDEOMAE_SMOKE_REPORT.md"
EXPECTED_OUTPUT_ROOT = PROJECT_ROOT / "data" / "importance_shards" / "bair_videomae_teacher_smoke"


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


def _resource_warning(resource: dict[str, Any], limits: dict[str, Any]) -> bool:
    ram_limit = float(limits.get("max_ram_gb_warning", 0.0) or 0.0)
    gpu_fraction_limit = float(limits.get("max_gpu_mem_fraction_warning", 0.0) or 0.0)
    ram_used = resource["ram_after"].get("used_gib")
    gpu_after = resource["gpu_memory_after"]
    gpu_used = gpu_after.get("used_gib")
    gpu_total = gpu_after.get("total_gib")
    ram_warn = bool(ram_limit and ram_used is not None and float(ram_used) > ram_limit)
    gpu_warn = bool(
        gpu_fraction_limit
        and gpu_used is not None
        and gpu_total not in (None, 0)
        and float(gpu_used) / float(gpu_total) > gpu_fraction_limit
    )
    return ram_warn or gpu_warn


def _all_finite(shard: dict[str, Any]) -> bool:
    tensor_keys = ("importance_scores", "importance_scores_norm", "base_losses", "masked_losses")
    return all(torch.isfinite(shard[key]).all().item() for key in tensor_keys)


def _finite_stat_block(summary: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return all(math.isfinite(float(summary[key])) for key in keys)


def _write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Importance BAIR VideoMAE Smoke Report",
        "",
        "Command: `python scripts/smoke_test_predictive_importance_bair_videomae.py`",
        "",
        f"- train token shard dir: `{summary['train_token_shard_dir']}`",
        f"- test token shard dir: `{summary['test_token_shard_dir']}`",
        f"- teacher checkpoint: `{summary['teacher_checkpoint']}`",
        f"- output root: `{summary['output_root']}`",
        f"- train samples: `{summary.get('train_samples')}`",
        f"- test samples: `{summary.get('test_samples')}`",
        f"- token shape: `{summary.get('token_shape')}`",
        f"- source encoder: `{summary.get('source_encoder')}`",
        f"- token_chunk_size: `{summary.get('token_chunk_size')}`",
        f"- train importance shard count: `{summary.get('train_shard_count')}`",
        f"- test importance shard count: `{summary.get('test_shard_count')}`",
        f"- GPU memory before: `{summary['resource']['gpu_memory_before']}`",
        f"- GPU memory after: `{summary['resource']['gpu_memory_after']}`",
        f"- RAM before: `{summary['resource']['ram_before']}`",
        f"- RAM after: `{summary['resource']['ram_after']}`",
        f"- elapsed time sec: `{summary['resource']['elapsed_time_sec']}`",
        f"- OOM: `{summary['resource']['oom']}`",
        f"- cloud recommendation: `{summary['cloud_recommendation']}`",
        f"- error: `{summary.get('error')}`",
        "",
        "## Checks",
        "",
        "```json",
        json.dumps(summary["checks"], indent=2),
        "```",
        "",
        "## First Train Shard Summary",
        "",
        "```json",
        json.dumps(summary.get("first_train_shard_summary", {}), indent=2),
        "```",
        "",
        "## First Test Shard Summary",
        "",
        "```json",
        json.dumps(summary.get("first_test_shard_summary", {}), indent=2),
        "```",
        "",
        "## Eval Summary",
        "",
        "```json",
        json.dumps(summary.get("eval_summary", {}), indent=2),
        "```",
        "",
        "## Generation Summary Digest",
        "",
        "```json",
        json.dumps(summary.get("generation_summary_digest", {}), indent=2),
        "```",
        "",
        f"IMPORTANCE_BAIR_VIDEOMAE_SMOKE_PASS = {str(summary['pass']).lower()}",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_bair_videomae_importance_smoke(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    train_token_dir = Path(config["data"]["train_token_shard_dir"])
    test_token_dir = Path(config["data"]["test_token_shard_dir"])
    token_glob = str(config["data"].get("shard_glob", "tokens_shard_*.pt"))
    train_token_paths = sorted(train_token_dir.glob(token_glob))
    test_token_paths = sorted(test_token_dir.glob(token_glob))
    if not train_token_paths:
        raise FileNotFoundError(
            f"Missing BAIR train VideoMAE token shards in {train_token_dir}; run Step 11B first."
        )
    if not test_token_paths:
        raise FileNotFoundError(
            f"Missing BAIR test VideoMAE token shards in {test_token_dir}; run Step 11B first."
        )

    teacher_checkpoint = Path(config["teacher"]["checkpoint"])
    if not teacher_checkpoint.exists():
        raise FileNotFoundError(f"Missing BAIR Teacher checkpoint {teacher_checkpoint}; run Step 11C first.")

    output_root = Path(config["output"]["output_root"])
    if output_root.resolve() != EXPECTED_OUTPUT_ROOT.resolve():
        raise ValueError(f"Refusing to delete unexpected output root: {output_root}")
    if output_root.exists():
        shutil.rmtree(output_root)

    first_train_token_shard = load_token_shard(train_token_paths[0], map_location="cpu")
    first_test_token_shard = load_token_shard(test_token_paths[0], map_location="cpu")
    source_encoder = str(first_train_token_shard.get("encoder_name", "unknown"))
    token_shape = list(first_train_token_shard["past_tokens"].shape)
    limits = dict(config.get("resource_limits", {}))
    ram_before = _ram_summary()
    gpu_before = _gpu_memory_gib()
    start_time = time.perf_counter()
    oom = False
    error: str | None = None
    generation_summary: dict[str, Any] | None = None
    eval_summary: dict[str, Any] | None = None
    first_train_summary: dict[str, Any] | None = None
    first_test_summary: dict[str, Any] | None = None
    first_train_inspect: dict[str, Any] | None = None
    first_test_inspect: dict[str, Any] | None = None
    train_shard_paths: list[Path] = []
    test_shard_paths: list[Path] = []
    train_finite_ok = False
    test_finite_ok = False

    try:
        generation_summary = generate_predictive_importance(config)
        train_output_dir = Path(config["output"]["train_output_dir"])
        test_output_dir = Path(config["output"]["test_output_dir"])
        train_shard_paths = sorted(train_output_dir.glob("importance_shard_*.pt"))
        test_shard_paths = sorted(test_output_dir.glob("importance_shard_*.pt"))
        first_train_shard = load_importance_shard(train_shard_paths[0], map_location="cpu")
        first_test_shard = load_importance_shard(test_shard_paths[0], map_location="cpu")
        validate_importance_shard(first_train_shard, strict=True)
        validate_importance_shard(first_test_shard, strict=True)
        first_train_summary = summarize_importance_shard(first_train_shard)
        first_test_summary = summarize_importance_shard(first_test_shard)
        first_train_inspect = inspect_importance_shard(train_shard_paths[0])
        first_test_inspect = inspect_importance_shard(test_shard_paths[0])
        eval_summary = evaluate_bair_importance_root(output_root)
        train_finite_ok = _all_finite(first_train_shard)
        test_finite_ok = _all_finite(first_test_shard)
    except Exception as exc:  # pragma: no cover - hardware dependent
        oom = _is_oom_error(exc)
        if oom and torch.cuda.is_available():
            torch.cuda.empty_cache()
        error = str(exc)

    elapsed = round(time.perf_counter() - start_time, 3)
    ram_after = _ram_summary()
    gpu_after = _gpu_memory_gib()
    resource = {
        "gpu_memory_before": gpu_before,
        "gpu_memory_after": gpu_after,
        "ram_before": ram_before,
        "ram_after": ram_after,
        "elapsed_time_sec": elapsed,
        "oom": oom,
    }
    resource_warn = _resource_warning(resource, limits)
    train_shard_count = len(train_shard_paths)
    test_shard_count = len(test_shard_paths)
    expected_train = int(config["data"]["max_train_samples"])
    expected_test = int(config["data"]["max_test_samples"])
    expected_tokens = 392

    checks = {
        "train_token_shards_exist": bool(train_token_paths),
        "test_token_shards_exist": bool(test_token_paths),
        "teacher_checkpoint_exists": teacher_checkpoint.exists(),
        "generation_summary_exists": bool(generation_summary),
        "train_generated_shards_ok": train_shard_count > 0,
        "test_generated_shards_ok": test_shard_count > 0,
        "train_importance_scores_shape_ok": bool(
            first_train_summary and first_train_summary["importance_scores_shape"] == [1, expected_tokens]
        ),
        "test_importance_scores_shape_ok": bool(
            first_test_summary and first_test_summary["importance_scores_shape"] == [1, expected_tokens]
        ),
        "train_norm_shape_ok": bool(
            first_train_summary and first_train_summary["importance_scores_norm_shape"] == [1, expected_tokens]
        ),
        "test_norm_shape_ok": bool(
            first_test_summary and first_test_summary["importance_scores_norm_shape"] == [1, expected_tokens]
        ),
        "train_split_ok": bool(first_train_summary and first_train_summary["split"] == "train"),
        "test_split_ok": bool(first_test_summary and first_test_summary["split"] == "test"),
        "train_inspect_matches_first": bool(
            first_train_inspect
            and first_train_summary
            and first_train_inspect["importance_scores_shape"] == first_train_summary["importance_scores_shape"]
        ),
        "test_inspect_matches_first": bool(
            first_test_inspect
            and first_test_summary
            and first_test_inspect["importance_scores_shape"] == first_test_summary["importance_scores_shape"]
        ),
        "train_finite_tensors_ok": train_finite_ok,
        "test_finite_tensors_ok": test_finite_ok,
        "eval_train_samples_ok": bool(
            eval_summary and eval_summary["split_summaries"]["train"]["num_samples"] == expected_train
        ),
        "eval_test_samples_ok": bool(
            eval_summary and eval_summary["split_summaries"]["test"]["num_samples"] == expected_test
        ),
        "eval_overall_samples_ok": bool(eval_summary and eval_summary["num_samples"] == expected_train + expected_test),
        "eval_tokens_ok": bool(eval_summary and eval_summary["num_tokens"] == expected_tokens),
        "eval_stats_finite_ok": bool(
            eval_summary
            and _finite_stat_block(
                eval_summary,
                (
                    "importance_mean",
                    "importance_std",
                    "importance_min",
                    "importance_max",
                    "normalized_importance_mean",
                    "normalized_importance_std",
                    "base_loss_mean",
                    "masked_loss_mean",
                ),
            )
        ),
        "oom_ok": not oom,
        "resource_warning_ok": not resource_warn,
    }
    pass_flag = all(checks.values())
    cloud_recommendation = (
        "recommended: OOM or resource warning; lower token_chunk_size or use 4090 / 48GB"
        if oom or resource_warn
        else "not required for Step 11D; local RTX 5080 smoke succeeded"
    )
    generation_digest = {}
    if generation_summary:
        generation_digest = {
            "num_samples": generation_summary.get("num_samples"),
            "train_num_samples": generation_summary.get("train_num_samples"),
            "test_num_samples": generation_summary.get("test_num_samples"),
            "num_tokens": generation_summary.get("num_tokens"),
            "token_dim": generation_summary.get("token_dim"),
            "num_importance_shards": generation_summary.get("num_importance_shards"),
            "token_chunk_size": generation_summary.get("token_chunk_size"),
            "elapsed_time_sec": generation_summary.get("elapsed_time_sec"),
            "oom": generation_summary.get("oom"),
        }

    summary = {
        "train_token_shard_dir": str(train_token_dir),
        "test_token_shard_dir": str(test_token_dir),
        "teacher_checkpoint": str(teacher_checkpoint),
        "output_root": str(output_root),
        "source_encoder": source_encoder,
        "used_fallback": False,
        "train_samples": expected_train,
        "test_samples": expected_test,
        "train_source_token_shape": list(first_train_token_shard["past_tokens"].shape),
        "test_source_token_shape": list(first_test_token_shard["past_tokens"].shape),
        "token_shape": token_shape,
        "token_chunk_size": int(config["importance"]["token_chunk_size"]),
        "batch_size": int(config["importance"]["batch_size"]),
        "train_shard_count": train_shard_count,
        "test_shard_count": test_shard_count,
        "first_train_shard_summary": first_train_summary,
        "first_test_shard_summary": first_test_summary,
        "eval_summary": eval_summary,
        "generation_summary_digest": generation_digest,
        "resource": resource,
        "cloud_recommendation": cloud_recommendation,
        "checks": checks,
        "pass": pass_flag,
        "error": error,
    }
    _write_report(summary)
    return summary


def main() -> None:
    summary = run_bair_videomae_importance_smoke()
    print(json.dumps(summary, indent=2))
    print(f"IMPORTANCE_BAIR_VIDEOMAE_SMOKE_PASS = {str(summary['pass']).lower()}")
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
