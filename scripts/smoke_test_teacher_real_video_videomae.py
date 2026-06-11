"""Smoke test TeacherWorldModel training on Step 9C real VideoMAE token shards."""

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

from eval.eval_teacher_prediction import evaluate_teacher
from scripts.inspect_teacher_checkpoint import inspect_checkpoint
from scripts.smoke_test_real_video_videomae_token_extraction import run_real_video_videomae_token_smoke
from training.train_teacher import load_yaml
from training.teacher_trainer import run_teacher_training


CONFIG_PATH = PROJECT_ROOT / "configs" / "train_teacher_real_video_videomae_smoke.yaml"
REPORT_PATH = PROJECT_ROOT / "docs" / "TEACHER_REAL_VIDEOMAE_SMOKE_REPORT.md"


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


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_oom_error(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Teacher Real VideoMAE Smoke Report",
        "",
        "Command: `python scripts/smoke_test_teacher_real_video_videomae.py`",
        "",
        f"- input token shard dir: `{summary['input_token_shard_dir']}`",
        f"- source encoder: `{summary.get('source_encoder')}`",
        f"- token shape: `{summary.get('token_shape')}`",
        f"- run dir: `{summary['run_dir']}`",
        f"- checkpoint path: `{summary.get('checkpoint_path')}`",
        f"- num steps: `{summary.get('num_steps')}`",
        f"- initial loss: `{summary.get('initial_loss')}`",
        f"- final loss: `{summary.get('final_loss')}`",
        f"- best loss: `{summary.get('best_loss')}`",
        f"- loss_decreased: `{summary.get('loss_decreased')}`",
        f"- eval mse: `{summary.get('eval_mse')}`",
        f"- GPU memory before: `{summary['resource']['gpu_memory_before']}`",
        f"- GPU memory after: `{summary['resource']['gpu_memory_after']}`",
        f"- RAM before: `{summary['resource']['ram_before']}`",
        f"- RAM after: `{summary['resource']['ram_after']}`",
        f"- elapsed time sec: `{summary['resource']['elapsed_time_sec']}`",
        f"- OOM: `{summary['resource']['oom']}`",
        f"- cloud recommendation: `{summary['cloud_recommendation']}`",
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
        f"TEACHER_REAL_VIDEOMAE_SMOKE_PASS = {str(summary['pass']).lower()}",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_teacher_real_videomae_smoke(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(config_path)
    shard_dir = Path(config["data"]["token_shard_dir"])
    shard_glob = str(config["data"].get("shard_glob", "tokens_shard_*.pt"))
    if not list(shard_dir.glob(shard_glob)):
        run_real_video_videomae_token_smoke()

    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    if run_dir.exists():
        shutil.rmtree(run_dir)

    ram_before = _ram_summary()
    gpu_before = _gpu_memory_gib()
    start_time = time.perf_counter()
    oom = False
    error: str | None = None
    training_summary: dict[str, Any] | None = None
    eval_summary: dict[str, Any] | None = None
    inspection: dict[str, Any] | None = None
    try:
        training_summary = run_teacher_training(config)
        checkpoint_path = Path(training_summary["checkpoint_path"])
        eval_summary = evaluate_teacher(config, checkpoint_path)
        inspection = inspect_checkpoint(checkpoint_path)
    except Exception as exc:  # pragma: no cover - hardware dependent
        oom = _is_oom_error(exc)
        error = str(exc)

    elapsed = round(time.perf_counter() - start_time, 3)
    ram_after = _ram_summary()
    gpu_after = _gpu_memory_gib()

    checkpoint_path = training_summary.get("checkpoint_path") if training_summary else None
    metrics_path = training_summary.get("metrics_path") if training_summary else None
    summary_path = training_summary.get("summary_path") if training_summary else None
    num_steps = int(training_summary.get("num_steps", 0)) if training_summary else 0
    loss_decreased = bool(training_summary.get("loss_decreased", False)) if training_summary else False
    checks = {
        "summary_exists": bool(summary_path and Path(summary_path).exists()),
        "metrics_exists": bool(metrics_path and Path(metrics_path).exists()),
        "checkpoint_exists": bool(checkpoint_path and Path(checkpoint_path).exists()),
        "num_steps_ok": num_steps >= 20,
        "initial_loss_finite": bool(training_summary and _finite(training_summary.get("initial_loss"))),
        "final_loss_finite": bool(training_summary and _finite(training_summary.get("final_loss"))),
        "best_loss_finite": bool(training_summary and _finite(training_summary.get("best_loss"))),
        "loss_decreased": loss_decreased,
        "eval_mse_finite": bool(eval_summary and _finite(eval_summary.get("eval_mse"))),
        "parameter_count_ok": bool(inspection and int(inspection.get("parameter_count", 0)) > 0),
        "oom_ok": not oom,
    }
    pass_flag = all(checks.values())
    summary = {
        "input_token_shard_dir": str(shard_dir),
        "run_dir": str(run_dir),
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "num_steps": num_steps,
        "initial_loss": training_summary.get("initial_loss") if training_summary else None,
        "final_loss": training_summary.get("final_loss") if training_summary else None,
        "best_loss": training_summary.get("best_loss") if training_summary else None,
        "loss_decreased": loss_decreased,
        "eval_mse": eval_summary.get("eval_mse") if eval_summary else None,
        "source_encoder": training_summary.get("source_encoder") if training_summary else None,
        "token_shape": [
            training_summary.get("num_tokens"),
            training_summary.get("token_dim"),
        ]
        if training_summary
        else None,
        "training_summary": training_summary,
        "eval_summary": eval_summary,
        "checkpoint_inspection": inspection,
        "resource": {
            "gpu_memory_before": gpu_before,
            "gpu_memory_after": gpu_after,
            "ram_before": ram_before,
            "ram_after": ram_after,
            "elapsed_time_sec": elapsed,
            "oom": oom,
        },
        "cloud_recommendation": (
            "recommended before Step 10B/11" if oom else "not required for Step 10A; local RTX 5080 smoke succeeded"
        ),
        "checks": checks,
        "pass": pass_flag,
        "error": error,
    }
    _write_report(summary)
    return summary


def main() -> None:
    summary = run_teacher_real_videomae_smoke()
    print(json.dumps(summary, indent=2))
    print(f"TEACHER_REAL_VIDEOMAE_SMOKE_PASS = {str(summary['pass']).lower()}")
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
