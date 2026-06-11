"""Smoke test predictive importance on Step 9C real VideoMAE token shards."""

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
from eval.eval_importance_summary import evaluate_importance_dir
from scripts.generate_predictive_importance import generate_predictive_importance, load_yaml
from scripts.inspect_importance_shard import inspect_importance_shard
from scripts.smoke_test_teacher_real_video_videomae import run_teacher_real_videomae_smoke


CONFIG_PATH = PROJECT_ROOT / "configs" / "generate_importance_real_video_videomae_smoke.yaml"
REPORT_PATH = PROJECT_ROOT / "docs" / "IMPORTANCE_REAL_VIDEOMAE_SMOKE_REPORT.md"
EXPECTED_OUTPUT_DIR = PROJECT_ROOT / "data" / "importance_shards" / "real_video_videomae_teacher_smoke"


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


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _all_finite(shard: dict[str, Any]) -> bool:
    tensor_keys = ("importance_scores", "importance_scores_norm", "base_losses", "masked_losses")
    return all(torch.isfinite(shard[key]).all().item() for key in tensor_keys)


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


def _write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Importance Real VideoMAE Smoke Report",
        "",
        "Command: `python scripts/smoke_test_predictive_importance_real_video_videomae.py`",
        "",
        f"- input token shard dir: `{summary['input_token_shard_dir']}`",
        f"- teacher checkpoint: `{summary['teacher_checkpoint']}`",
        f"- output importance dir: `{summary['output_importance_dir']}`",
        f"- num samples: `{summary.get('num_samples')}`",
        f"- token shape: `{summary.get('token_shape')}`",
        f"- token_chunk_size: `{summary.get('token_chunk_size')}`",
        f"- generated shard count: `{summary.get('generated_shard_count')}`",
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
        "## First Shard Summary",
        "",
        "```json",
        json.dumps(summary.get("first_shard_summary", {}), indent=2),
        "```",
        "",
        "## Eval Summary",
        "",
        "```json",
        json.dumps(summary.get("eval_summary", {}), indent=2),
        "```",
        "",
        "## Generation Summary",
        "",
        "```json",
        json.dumps(summary.get("generation_summary", {}), indent=2),
        "```",
        "",
        f"IMPORTANCE_REAL_VIDEOMAE_SMOKE_PASS = {str(summary['pass']).lower()}",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_real_videomae_importance_smoke(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    token_dir = Path(config["data"]["token_shard_dir"])
    token_glob = str(config["data"].get("shard_glob", "tokens_shard_*.pt"))
    token_shard_paths = sorted(token_dir.glob(token_glob))
    if not token_shard_paths:
        raise FileNotFoundError(
            f"Missing real VideoMAE token shards in {token_dir}; Step 10B does not rerun extraction."
        )

    teacher_checkpoint = Path(config["teacher"]["checkpoint"])
    if not teacher_checkpoint.exists():
        run_teacher_real_videomae_smoke()
    if not teacher_checkpoint.exists():
        raise FileNotFoundError(f"Missing teacher checkpoint: {teacher_checkpoint}")

    output_dir = Path(config["output"]["output_dir"])
    if output_dir.resolve() != EXPECTED_OUTPUT_DIR.resolve():
        raise ValueError(f"Refusing to delete unexpected output dir: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    first_token_shard = load_token_shard(token_shard_paths[0], map_location="cpu")
    source_encoder = str(first_token_shard.get("encoder_name", "unknown"))
    token_shape = list(first_token_shard["past_tokens"].shape)
    limits = dict(config.get("resource_limits", {}))
    ram_before = _ram_summary()
    gpu_before = _gpu_memory_gib()
    start_time = time.perf_counter()
    oom = False
    error: str | None = None
    generation_summary: dict[str, Any] | None = None
    eval_summary: dict[str, Any] | None = None
    first_summary: dict[str, Any] | None = None
    inspect_summary: dict[str, Any] | None = None

    try:
        generation_summary = generate_predictive_importance(config)
        shard_paths = sorted(output_dir.glob("importance_shard_*.pt"))
        first_shard = load_importance_shard(shard_paths[0], map_location="cpu")
        validate_importance_shard(first_shard, strict=True)
        first_summary = summarize_importance_shard(first_shard)
        inspect_summary = inspect_importance_shard(shard_paths[0])
        eval_summary = evaluate_importance_dir(output_dir)
        finite_ok = _all_finite(first_shard)
    except Exception as exc:  # pragma: no cover - hardware dependent
        oom = _is_oom_error(exc)
        if oom and torch.cuda.is_available():
            torch.cuda.empty_cache()
        error = str(exc)
        shard_paths = sorted(output_dir.glob("importance_shard_*.pt")) if output_dir.exists() else []
        finite_ok = False

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
    generated_count = len(shard_paths)
    checks = {
        "token_shards_exist": bool(token_shard_paths),
        "teacher_checkpoint_exists": teacher_checkpoint.exists(),
        "generation_summary_exists": bool(generation_summary),
        "generated_shards_ok": generated_count >= 1,
        "importance_scores_shape_ok": bool(first_summary and first_summary["importance_scores_shape"][1] == 784),
        "importance_scores_norm_shape_ok": bool(
            first_summary and first_summary["importance_scores_norm_shape"][1] == 784
        ),
        "base_losses_shape_ok": bool(first_summary and len(first_summary["base_losses_shape"]) == 1),
        "masked_losses_shape_ok": bool(first_summary and first_summary["masked_losses_shape"][1] == 784),
        "inspect_matches_first": bool(
            inspect_summary
            and first_summary
            and inspect_summary["importance_scores_shape"] == first_summary["importance_scores_shape"]
        ),
        "eval_samples_ok": bool(eval_summary and int(eval_summary["num_samples"]) == 8),
        "eval_tokens_ok": bool(eval_summary and int(eval_summary["num_tokens"]) == 784),
        "finite_tensors_ok": finite_ok,
        "oom_ok": not oom,
        "resource_warning_ok": not resource_warn,
    }
    pass_flag = all(checks.values())
    cloud_recommendation = (
        "recommended: OOM or resource warning; lower token_chunk_size or use 4090 / 48GB"
        if oom or resource_warn
        else "not required for Step 10B; local RTX 5080 smoke succeeded"
    )
    summary = {
        "input_token_shard_dir": str(token_dir),
        "teacher_checkpoint": str(teacher_checkpoint),
        "output_importance_dir": str(output_dir),
        "source_encoder": source_encoder,
        "used_fallback": False,
        "num_samples": int(generation_summary.get("num_samples", 0)) if generation_summary else None,
        "token_shape": token_shape,
        "token_chunk_size": int(config["importance"]["token_chunk_size"]),
        "batch_size": int(config["importance"]["batch_size"]),
        "generated_shard_count": generated_count,
        "first_shard_summary": first_summary,
        "eval_summary": eval_summary,
        "generation_summary": generation_summary,
        "resource": resource,
        "cloud_recommendation": cloud_recommendation,
        "checks": checks,
        "pass": pass_flag,
        "error": error,
    }
    _write_report(summary)
    return summary


def main() -> None:
    summary = run_real_videomae_importance_smoke()
    print(json.dumps(summary, indent=2))
    print(f"IMPORTANCE_REAL_VIDEOMAE_SMOKE_PASS = {str(summary['pass']).lower()}")
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
