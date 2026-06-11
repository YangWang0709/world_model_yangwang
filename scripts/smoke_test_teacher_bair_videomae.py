"""Smoke train/eval a TeacherWorldModel on BAIR VideoMAE token shards."""

from __future__ import annotations

import json
import math
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

from eval.eval_teacher_bair_videomae import evaluate_teacher
from scripts.inspect_teacher_checkpoint import inspect_checkpoint
from training.train_teacher import load_yaml
from training.teacher_trainer import run_teacher_training


CONFIG_PATH = PROJECT_ROOT / "configs" / "train_teacher_bair_videomae_smoke.yaml"
REPORT_PATH = PROJECT_ROOT / "docs" / "TEACHER_BAIR_VIDEOMAE_SMOKE_REPORT.md"


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


def _safe_clear_run_dir(run_dir: Path) -> None:
    expected = (PROJECT_ROOT / "runs" / "teacher_bair_videomae_smoke_v1").resolve()
    resolved = run_dir.resolve()
    if resolved != expected:
        raise ValueError(f"Refusing to delete unexpected run dir: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _count_shards(path: Path) -> int:
    return len(sorted(path.glob("tokens_shard_*.pt")))


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                metrics.append(json.loads(stripped))
    return metrics


def _assert_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise FloatingPointError(f"{name} is not finite: {value}")


def _write_report(result: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Teacher BAIR VideoMAE Smoke Report",
        "",
        f"- command: `{result['command']}`",
        f"- train token shard dir: `{result['train_token_shard_dir']}`",
        f"- test token shard dir: `{result['test_token_shard_dir']}`",
        f"- train samples: `{result['train_num_samples']}`",
        f"- test samples: `{result['test_num_samples']}`",
        f"- token shape: `{result['token_shape']}`",
        f"- run dir: `{result['run_dir']}`",
        f"- checkpoint path: `{result['checkpoint_path']}`",
        f"- num steps: `{result['num_steps']}`",
        f"- initial loss: `{result['initial_loss']}`",
        f"- final loss: `{result['final_loss']}`",
        f"- best loss: `{result['best_loss']}`",
        f"- loss_decreased: `{result['loss_decreased']}`",
        f"- loss_warning: `{result['loss_warning']}`",
        f"- eval_mse: `{result['eval_mse']}`",
        f"- elapsed_time_sec: `{result['elapsed_time_sec']}`",
        f"- oom: `{result['oom']}`",
        f"- cloud_recommendation: `{result['cloud_recommendation']}`",
        f"- TEACHER_BAIR_VIDEOMAE_SMOKE_PASS: `{str(result['smoke_pass']).lower()}`",
        "",
        "## Resource Before",
        "",
        "```json",
        json.dumps(result["resource_before"], indent=2, sort_keys=True),
        "```",
        "",
        "## Resource After",
        "",
        "```json",
        json.dumps(result["resource_after"], indent=2, sort_keys=True),
        "```",
        "",
        "## Training Summary",
        "",
        "```json",
        json.dumps(result["training_summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Eval Summary",
        "",
        "```json",
        json.dumps(result["eval_summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Checkpoint Summary",
        "",
        "```json",
        json.dumps(result["checkpoint_summary"], indent=2, sort_keys=True),
        "```",
        "",
        "TEACHER_BAIR_VIDEOMAE_SMOKE_PASS = true",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    config = load_yaml(CONFIG_PATH)
    data_cfg = config["data"]
    output_cfg = config["output"]
    train_dir = Path(data_cfg["train_token_shard_dir"])
    test_dir = Path(data_cfg["test_token_shard_dir"])
    run_dir = Path(output_cfg["run_root"]) / output_cfg["run_name"]

    train_shards = _count_shards(train_dir)
    test_shards = _count_shards(test_dir)
    if train_shards <= 0 or test_shards <= 0:
        raise FileNotFoundError(
            "Step 11C requires existing BAIR VideoMAE token shards from Step 11B. "
            "Do not download data or rerun VideoMAE here; run Step 11B first if shards are missing."
        )

    _safe_clear_run_dir(run_dir)
    resource_before = _resource_snapshot()
    oom = False
    try:
        training_summary = run_teacher_training(config)
        checkpoint_path = Path(training_summary["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint missing: {checkpoint_path}")
        metrics_path = Path(training_summary["metrics_path"])
        if not metrics_path.exists():
            raise FileNotFoundError(f"Teacher metrics missing: {metrics_path}")
        if not Path(training_summary["summary_path"]).exists():
            raise FileNotFoundError(f"Teacher summary missing: {training_summary['summary_path']}")
        eval_summary = evaluate_teacher(config, checkpoint_path)
        checkpoint_summary = inspect_checkpoint(checkpoint_path)
    except RuntimeError as exc:
        message = str(exc).lower()
        oom = "out of memory" in message or "oom" in message
        if oom:
            raise RuntimeError(
                "BAIR Teacher smoke hit OOM with conservative settings. "
                "Retry with batch_size=2 or use a cloud 4090 / 48GB GPU server."
            ) from exc
        raise

    metrics = _load_metrics(Path(training_summary["metrics_path"]))
    if int(training_summary["num_steps"]) < 50:
        raise AssertionError(f"Expected at least 50 training steps, got {training_summary['num_steps']}")
    for key in ("initial_loss", "final_loss", "best_loss"):
        _assert_finite(key, float(training_summary[key]))
    _assert_finite("eval_mse", float(eval_summary["eval_mse"]))

    loss_decreased = bool(training_summary["loss_decreased"])
    loss_warning = None if loss_decreased else "final_loss did not fall below initial_loss on this tiny smoke"
    token_shape = [
        int(training_summary["train_num_samples"]),
        int(training_summary["num_tokens"]),
        int(training_summary["token_dim"]),
    ]
    resource_after = _resource_snapshot()
    result = {
        "command": (
            "/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab "
            "python scripts/smoke_test_teacher_bair_videomae.py"
        ),
        "train_token_shard_dir": str(train_dir),
        "test_token_shard_dir": str(test_dir),
        "train_shards": train_shards,
        "test_shards": test_shards,
        "train_num_samples": int(training_summary["train_num_samples"]),
        "test_num_samples": int(eval_summary["num_samples"]),
        "token_shape": token_shape,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "num_steps": int(training_summary["num_steps"]),
        "initial_loss": float(training_summary["initial_loss"]),
        "final_loss": float(training_summary["final_loss"]),
        "best_loss": float(training_summary["best_loss"]),
        "loss_decreased": loss_decreased,
        "loss_warning": loss_warning,
        "eval_mse": float(eval_summary["eval_mse"]),
        "metrics_rows": len(metrics),
        "training_summary": training_summary,
        "eval_summary": eval_summary,
        "checkpoint_summary": checkpoint_summary,
        "resource_before": resource_before,
        "resource_after": resource_after,
        "elapsed_time_sec": round(time.perf_counter() - start, 3),
        "oom": oom,
        "cloud_recommendation": "not_needed_for_step11c",
        "smoke_pass": True,
    }
    _write_report(result)
    print(REPORT_PATH.read_text(encoding="utf-8"))
    print("TEACHER_BAIR_VIDEOMAE_SMOKE_PASS = true")


if __name__ == "__main__":
    main()
