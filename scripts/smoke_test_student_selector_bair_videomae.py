"""Smoke test Student selector training on BAIR VideoMAE importance labels."""

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

from eval.eval_student_selector import evaluate_student_selector
from scripts.inspect_student_selector_checkpoint import inspect_student_selector_checkpoint
from training.student_selector_trainer import run_student_selector_training
from training.train_student_selector import load_yaml


CONFIG_PATH = PROJECT_ROOT / "configs" / "train_student_selector_bair_videomae_smoke.yaml"
REPORT_PATH = PROJECT_ROOT / "docs" / "STUDENT_SELECTOR_BAIR_VIDEOMAE_SMOKE_REPORT.md"
EXPECTED_RUN_DIR = PROJECT_ROOT / "runs" / "student_selector_bair_videomae_smoke_v1"


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
        "# Student Selector BAIR VideoMAE Smoke Report",
        "",
        "Command: `python scripts/smoke_test_student_selector_bair_videomae.py`",
        "",
        f"- train token shard dir: `{summary['train_token_shard_dir']}`",
        f"- test token shard dir: `{summary['test_token_shard_dir']}`",
        f"- train importance shard dir: `{summary['train_importance_shard_dir']}`",
        f"- test importance shard dir: `{summary['test_importance_shard_dir']}`",
        f"- run dir: `{summary['run_dir']}`",
        f"- checkpoint path: `{summary.get('checkpoint_path')}`",
        f"- train samples: `{summary.get('train_samples')}`",
        f"- test samples: `{summary.get('test_samples')}`",
        f"- num tokens: `{summary.get('num_tokens')}`",
        f"- topk: `{summary.get('topk')}`",
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
        "## Training Summary",
        "",
        "```json",
        json.dumps(summary.get("training_summary", {}), indent=2),
        "```",
        "",
        "## Eval Summary",
        "",
        "```json",
        json.dumps(summary.get("eval_summary", {}), indent=2),
        "```",
        "",
        "## Checkpoint Inspection",
        "",
        "```json",
        json.dumps(summary.get("checkpoint_inspection", {}), indent=2),
        "```",
        "",
        f"STUDENT_SELECTOR_BAIR_VIDEOMAE_SMOKE_PASS = {str(summary['pass']).lower()}",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_student_selector_bair_videomae_smoke(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    data_cfg = config["data"]
    train_token_dir = Path(data_cfg["train_token_shard_dir"])
    test_token_dir = Path(data_cfg["test_token_shard_dir"])
    train_importance_dir = Path(data_cfg["train_importance_shard_dir"])
    test_importance_dir = Path(data_cfg["test_importance_shard_dir"])
    token_glob = str(data_cfg.get("token_shard_glob", "tokens_shard_*.pt"))
    importance_glob = str(data_cfg.get("importance_shard_glob", "importance_shard_*.pt"))
    if not list(train_token_dir.glob(token_glob)):
        raise FileNotFoundError(f"Missing BAIR train token shards in {train_token_dir}; run Step 11B first.")
    if not list(test_token_dir.glob(token_glob)):
        raise FileNotFoundError(f"Missing BAIR test token shards in {test_token_dir}; run Step 11B first.")
    if not list(train_importance_dir.glob(importance_glob)):
        raise FileNotFoundError(f"Missing BAIR train importance shards in {train_importance_dir}; run Step 11D first.")
    if not list(test_importance_dir.glob(importance_glob)):
        raise FileNotFoundError(f"Missing BAIR test importance shards in {test_importance_dir}; run Step 11D first.")

    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    if run_dir.resolve() != EXPECTED_RUN_DIR.resolve():
        raise ValueError(f"Refusing to delete unexpected run dir: {run_dir}")
    if run_dir.exists():
        shutil.rmtree(run_dir)

    limits = dict(config.get("resource_limits", {}))
    ram_before = _ram_summary()
    gpu_before = _gpu_memory_gib()
    start_time = time.perf_counter()
    oom = False
    error: str | None = None
    training_summary: dict[str, Any] | None = None
    eval_summary: dict[str, Any] | None = None
    inspection: dict[str, Any] | None = None
    try:
        training_summary = run_student_selector_training(config)
        eval_summary = evaluate_student_selector(config, training_summary["checkpoint_path"], split="test")
        inspection = inspect_student_selector_checkpoint(training_summary["checkpoint_path"])
    except Exception as exc:  # pragma: no cover - hardware dependent
        oom = _is_oom_error(exc)
        if oom and torch.cuda.is_available():
            torch.cuda.empty_cache()
        error = str(exc)

    elapsed = round(time.perf_counter() - start_time, 3)
    resource = {
        "gpu_memory_before": gpu_before,
        "gpu_memory_after": _gpu_memory_gib(),
        "ram_before": ram_before,
        "ram_after": _ram_summary(),
        "elapsed_time_sec": elapsed,
        "oom": oom,
    }
    resource_warn = _resource_warning(resource, limits)
    checkpoint_path = training_summary.get("checkpoint_path") if training_summary else None
    metrics_path = training_summary.get("metrics_path") if training_summary else None
    summary_path = training_summary.get("summary_path") if training_summary else None
    checks = {
        "train_token_shards_exist": bool(list(train_token_dir.glob(token_glob))),
        "test_token_shards_exist": bool(list(test_token_dir.glob(token_glob))),
        "train_importance_shards_exist": bool(list(train_importance_dir.glob(importance_glob))),
        "test_importance_shards_exist": bool(list(test_importance_dir.glob(importance_glob))),
        "summary_exists": bool(summary_path and Path(summary_path).exists()),
        "metrics_exists": bool(metrics_path and Path(metrics_path).exists()),
        "checkpoint_exists": bool(checkpoint_path and Path(checkpoint_path).exists()),
        "num_steps_ok": bool(training_summary and int(training_summary.get("num_steps", 0)) >= 50),
        "initial_loss_finite": bool(training_summary and _finite(training_summary.get("initial_loss"))),
        "final_loss_finite": bool(training_summary and _finite(training_summary.get("final_loss"))),
        "best_loss_finite": bool(training_summary and _finite(training_summary.get("best_loss"))),
        "loss_decreased": bool(training_summary and training_summary.get("loss_decreased") is True),
        "train_importance_mse_finite": bool(
            training_summary and _finite(training_summary.get("final_train_importance_mse"))
        ),
        "train_pearson_corr_finite": bool(
            training_summary and _finite(training_summary.get("final_train_pearson_corr_mean"))
        ),
        "train_target_topk_overlap_finite": bool(
            training_summary and _finite(training_summary.get("final_train_target_topk_overlap"))
        ),
        "test_importance_mse_finite": bool(training_summary and _finite(training_summary.get("test_importance_mse"))),
        "test_importance_mae_finite": bool(training_summary and _finite(training_summary.get("test_importance_mae"))),
        "test_pearson_corr_finite": bool(
            training_summary and _finite(training_summary.get("test_pearson_corr_mean"))
        ),
        "test_pearson_corr_positive": bool(
            training_summary and float(training_summary.get("test_pearson_corr_mean", 0.0)) > 0.0
        ),
        "test_target_top1_overlap_finite": bool(
            training_summary and _finite(training_summary.get("test_target_top1_overlap"))
        ),
        "test_target_topk_overlap_finite": bool(
            training_summary and _finite(training_summary.get("test_target_topk_overlap"))
        ),
        "test_target_topk_overlap_nonzero": bool(
            training_summary and float(training_summary.get("test_target_topk_overlap", 0.0)) > 0.0
        ),
        "test_selected_teacher_importance_finite": bool(
            training_summary and _finite(training_summary.get("test_selected_teacher_importance_mean"))
        ),
        "test_random_teacher_importance_finite": bool(
            training_summary and _finite(training_summary.get("test_random_teacher_importance_mean"))
        ),
        "test_selected_vs_random_gap_finite": bool(
            training_summary and _finite(training_summary.get("test_selected_vs_random_importance_gap"))
        ),
        "test_selected_vs_random_gap_positive": bool(
            training_summary and float(training_summary.get("test_selected_vs_random_importance_gap", 0.0)) > 0.0
        ),
        "eval_importance_mse_finite": bool(eval_summary and _finite(eval_summary.get("importance_mse"))),
        "eval_target_topk_overlap_finite": bool(eval_summary and _finite(eval_summary.get("target_topk_overlap"))),
        "parameter_count_ok": bool(inspection and int(inspection.get("parameter_count", 0)) > 0),
        "oom_ok": not oom,
        "resource_warning_ok": not resource_warn,
    }
    pass_flag = all(checks.values())
    if training_summary:
        training_summary["smoke_checks"] = checks
        if summary_path:
            Path(summary_path).write_text(json.dumps(training_summary, indent=2), encoding="utf-8")

    cloud_recommendation = (
        "recommended: OOM or resource warning; use 4090 / 48GB before scaling"
        if oom or resource_warn
        else "not required for Step 11E; local RTX 5080 smoke succeeded"
    )
    summary = {
        "train_token_shard_dir": str(train_token_dir),
        "test_token_shard_dir": str(test_token_dir),
        "train_importance_shard_dir": str(train_importance_dir),
        "test_importance_shard_dir": str(test_importance_dir),
        "run_dir": str(run_dir),
        "checkpoint_path": checkpoint_path,
        "train_samples": training_summary.get("train_num_samples") if training_summary else None,
        "test_samples": training_summary.get("test_num_samples") if training_summary else None,
        "num_tokens": training_summary.get("num_tokens") if training_summary else None,
        "topk": int(config["training"]["topk"]),
        "training_summary": training_summary,
        "eval_summary": eval_summary,
        "checkpoint_inspection": inspection,
        "resource": resource,
        "cloud_recommendation": cloud_recommendation,
        "checks": checks,
        "pass": pass_flag,
        "error": error,
    }
    _write_report(summary)
    return summary


def main() -> None:
    summary = run_student_selector_bair_videomae_smoke()
    print(json.dumps(summary, indent=2))
    print(f"STUDENT_SELECTOR_BAIR_VIDEOMAE_SMOKE_PASS = {str(summary['pass']).lower()}")
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
