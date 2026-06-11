"""End-to-end Step 11F smoke for compressed Student world model on BAIR VideoMAE tokens."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_student_world_model_bair_videomae import evaluate_student_world_model
from eval.eval_teacher_student_gap_bair_videomae import evaluate_teacher_student_gap
from scripts.inspect_student_world_model_checkpoint import inspect_student_world_model_checkpoint
from training.student_world_model_trainer import run_student_world_model_training
from training.train_student_world_model import load_yaml


CONFIG_PATH = PROJECT_ROOT / "configs" / "train_student_world_model_bair_videomae_smoke.yaml"
REPORT_PATH = PROJECT_ROOT / "docs" / "STUDENT_WORLD_MODEL_BAIR_VIDEOMAE_SMOKE_REPORT.md"
STEP11F_DOC_PATH = PROJECT_ROOT / "docs" / "STEP11F_STUDENT_WORLD_MODEL_BAIR_VIDEOMAE.md"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _ram_snapshot() -> dict[str, float]:
    values: dict[str, float] = {}
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return {"ram_used_gib": 0.0, "ram_total_gib": 0.0}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        key, raw_value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = float(raw_value.strip().split()[0]) * 1024.0
    total = values.get("MemTotal", 0.0)
    available = values.get("MemAvailable", 0.0)
    gib = 1024.0**3
    return {
        "ram_used_gib": (total - available) / gib if total else 0.0,
        "ram_total_gib": total / gib if total else 0.0,
    }


def _gpu_snapshot() -> dict[str, float | str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return {"gpu_name": "unavailable", "gpu_mem_total_gib": 0.0, "gpu_mem_used_gib": 0.0}
    first = result.stdout.strip().splitlines()[0]
    name, total_mib, used_mib = [part.strip() for part in first.split(",")]
    return {
        "gpu_name": name,
        "gpu_mem_total_gib": float(total_mib) / 1024.0,
        "gpu_mem_used_gib": float(used_mib) / 1024.0,
    }


def _resource_snapshot() -> dict[str, Any]:
    return {**_ram_snapshot(), **_gpu_snapshot()}


def _has_shards(path: Path, pattern: str) -> bool:
    return bool(list(path.glob(pattern)))


def _ensure_inputs(config: dict[str, Any]) -> None:
    data_cfg = config["data"]
    checks = [
        (Path(data_cfg["train_token_shard_dir"]), data_cfg.get("token_shard_glob", "tokens_shard_*.pt")),
        (Path(data_cfg["test_token_shard_dir"]), data_cfg.get("token_shard_glob", "tokens_shard_*.pt")),
        (Path(data_cfg["train_importance_shard_dir"]), data_cfg.get("importance_shard_glob", "importance_shard_*.pt")),
        (Path(data_cfg["test_importance_shard_dir"]), data_cfg.get("importance_shard_glob", "importance_shard_*.pt")),
    ]
    for shard_dir, pattern in checks:
        if not _has_shards(shard_dir, pattern):
            raise FileNotFoundError(f"Missing BAIR shards under {shard_dir} with pattern {pattern}")

    selector_checkpoint = Path(config["selector"]["checkpoint"])
    if not selector_checkpoint.exists():
        raise FileNotFoundError(
            "Missing Step 11E selector checkpoint; run Step 11E first. "
            f"This Step 11F smoke does not retrain the selector automatically: {selector_checkpoint}"
        )
    teacher_checkpoint = Path(config["teacher_reference"]["checkpoint"])
    if not teacher_checkpoint.exists():
        raise FileNotFoundError(f"Missing Step 11C teacher checkpoint: {teacher_checkpoint}")


def _clean_run_dir(config: dict[str, Any]) -> Path:
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    expected = (PROJECT_ROOT / "runs" / "student_world_model_bair_videomae_smoke_v1").resolve()
    resolved = run_dir.resolve()
    if resolved != expected:
        raise ValueError(f"Refusing to clean unexpected run dir: {run_dir}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    return run_dir


def _write_report(
    config: dict[str, Any],
    training_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    gap_summary: dict[str, Any],
    inspection: dict[str, Any],
    resources: dict[str, Any],
    checks: dict[str, bool],
    pass_flag: bool,
    elapsed_time_sec: float,
    oom: bool,
) -> None:
    command = (
        "/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab "
        "python scripts/smoke_test_student_world_model_bair_videomae.py"
    )
    data_cfg = config["data"]
    report = [
        "# Student World Model BAIR VideoMAE Smoke Report",
        "",
        f"Command: `{command}`",
        "",
        f"- train token shard dir: `{data_cfg['train_token_shard_dir']}`",
        f"- test token shard dir: `{data_cfg['test_token_shard_dir']}`",
        f"- train importance shard dir: `{data_cfg['train_importance_shard_dir']}`",
        f"- test importance shard dir: `{data_cfg['test_importance_shard_dir']}`",
        f"- selector checkpoint: `{training_summary.get('selector_checkpoint_path', config['selector']['checkpoint'])}`",
        f"- teacher checkpoint: `{gap_summary.get('teacher_checkpoint_path', config['teacher_reference']['checkpoint'])}`",
        f"- run dir: `{training_summary.get('run_dir', Path(config['output']['run_root']) / config['output']['run_name'])}`",
        f"- checkpoint path: `{training_summary.get('checkpoint_path', '')}`",
        f"- train samples: `{training_summary.get('train_num_samples')}`",
        f"- test samples: `{training_summary.get('test_num_samples')}`",
        f"- num tokens: `{training_summary.get('num_tokens')}`",
        f"- topk: `{training_summary.get('topk')}`",
        f"- token_retention_ratio: `{gap_summary.get('token_retention_ratio')}`",
        f"- elapsed_time_sec: `{elapsed_time_sec}`",
        f"- oom: `{str(oom).lower()}`",
        f"- cloud recommendation: `{resources.get('cloud_recommendation')}`",
        "",
        "## Training Summary",
        "",
        "```json",
        json.dumps(training_summary, indent=2),
        "```",
        "",
        "## Eval Summary",
        "",
        "```json",
        json.dumps(eval_summary, indent=2),
        "```",
        "",
        "## Teacher-Student Gap Summary",
        "",
        "```json",
        json.dumps(gap_summary, indent=2),
        "```",
        "",
        "## Checkpoint Inspection Summary",
        "",
        "```json",
        json.dumps(
            {
                "step": inspection.get("step"),
                "selector_frozen": inspection.get("selector_frozen"),
                "selector_config": inspection.get("selector_config"),
                "compressor_config": inspection.get("compressor_config"),
                "student_world_model_config": inspection.get("student_world_model_config"),
                "selector_parameter_count": inspection.get("selector_parameter_count"),
                "compressor_parameter_count": inspection.get("compressor_parameter_count"),
                "student_world_model_parameter_count": inspection.get("student_world_model_parameter_count"),
            },
            indent=2,
        ),
        "```",
        "",
        "## Resource Usage",
        "",
        "```json",
        json.dumps(resources, indent=2),
        "```",
        "",
        "## Checks",
        "",
        "```json",
        json.dumps(checks, indent=2),
        "```",
        "",
        f"STUDENT_WORLD_MODEL_BAIR_VIDEOMAE_SMOKE_PASS = {str(pass_flag).lower()}",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_step11f_doc(
    config: dict[str, Any],
    training_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    gap_summary: dict[str, Any],
    resources: dict[str, Any],
    pass_flag: bool,
    pytest_result: str = "Run after Step 11F smoke; final result is recorded in the final response/local summary.",
) -> None:
    data_cfg = config["data"]
    doc = [
        "# STEP11F Student World Model on BAIR VideoMAE Tokens",
        "",
        "## 1. Goal",
        "",
        "Step 11F connects the BAIR Student selector, `TokenCompressor`, and `StudentWorldModel` into a compressed future-prediction smoke pipeline on public BAIR Robot Pushing small clips represented as VideoMAE tokens.",
        "",
        "## 2. Cloud Server Decision",
        "",
        "No cloud server is required for this stage. The run reads existing BAIR VideoMAE token shards and predictive-importance shards, and it does not run the VideoMAE encoder.",
        "",
        f"- OOM: `{resources.get('oom', False)}`",
        f"- max RAM used GiB: `{resources.get('max_ram_used_gib')}`",
        f"- max GPU memory used GiB: `{resources.get('max_gpu_mem_used_gib')}`",
        f"- cloud recommendation: `{resources.get('cloud_recommendation')}`",
        "",
        "## 3. Input",
        "",
        f"- train token dir: `{data_cfg['train_token_shard_dir']}`",
        f"- test token dir: `{data_cfg['test_token_shard_dir']}`",
        f"- train importance dir: `{data_cfg['train_importance_shard_dir']}`",
        f"- test importance dir: `{data_cfg['test_importance_shard_dir']}`",
        f"- selector checkpoint: `{config['selector']['checkpoint']}`",
        f"- teacher checkpoint: `{config['teacher_reference']['checkpoint']}`",
        f"- train samples: `{training_summary.get('train_num_samples')}`",
        f"- test samples: `{training_summary.get('test_num_samples')}`",
        f"- token shape: `[B, {training_summary.get('num_tokens')}, {training_summary.get('token_dim')}]`",
        "- source encoder: `videomae`",
        "",
        "## 4. Pipeline",
        "",
        "`past_tokens -> frozen selector -> topK selected tokens -> TokenCompressor -> StudentWorldModel -> future latent prediction`.",
        "",
        "## 5. Training Setup",
        "",
        f"- selector frozen: `{training_summary.get('selector_frozen')}`",
        f"- topK: `{training_summary.get('topk')}`",
        f"- token_retention_ratio: `{training_summary.get('token_retention_ratio')}`",
        f"- compressed latents: `{training_summary.get('compressed_tokens')}`",
        "- target: mean-pooled `future_tokens` latent",
        "- optimized modules: `TokenCompressor` + `StudentWorldModel`",
        "- eval split: BAIR test",
        "",
        "## 6. Smoke Test Result",
        "",
        f"- initial_loss: `{training_summary.get('initial_loss')}`",
        f"- final_loss: `{training_summary.get('final_loss')}`",
        f"- best_loss: `{training_summary.get('best_loss')}`",
        f"- loss_decreased: `{training_summary.get('loss_decreased')}`",
        f"- student_future_mse: `{eval_summary.get('student_future_mse')}`",
        f"- selector_target_top1_overlap: `{eval_summary.get('selector_target_top1_overlap')}`",
        f"- selector_target_topK_overlap: `{eval_summary.get('selector_target_topk_overlap')}`",
        f"- selected_teacher_importance_mean: `{eval_summary.get('selected_teacher_importance_mean')}`",
        f"- STUDENT_WORLD_MODEL_BAIR_VIDEOMAE_SMOKE_PASS: `{str(pass_flag).lower()}`",
        "",
        "## 7. Teacher-Student Gap",
        "",
        f"- teacher_mse: `{gap_summary.get('teacher_mse')}`",
        f"- student_mse: `{gap_summary.get('student_mse')}`",
        f"- student_teacher_gap: `{gap_summary.get('student_teacher_gap')}`",
        f"- student_teacher_ratio: `{gap_summary.get('student_teacher_ratio')}`",
        f"- token_retention_ratio: `{gap_summary.get('token_retention_ratio')}`",
        "",
        "## 8. Pytest Result",
        "",
        pytest_result,
        "",
        "## 9. Git Commit",
        "",
        "Branch: `feature/tgpawb-step3-token-extraction`. The final commit hash and push status are recorded after commit/push in `STEP11F_LOCAL_SUMMARY.md` and the final response. No PR is created.",
        "",
        "## 10. What Was Not Done",
        "",
        "- no new model download",
        "- no new dataset download",
        "- no BAIR re-download",
        "- no VideoMAE training",
        "- no Teacher retraining",
        "- no selector retraining",
        "- no VLM grounding",
        "- no RL / policy optimization",
        "- no model weight committed",
        "- no BAIR data committed",
        "- no token shard committed",
        "- no importance shard committed",
        "- no checkpoint committed",
        "- no password or token saved",
        "- no PR created",
        "",
        "## 11. Next Step Recommendation",
        "",
        "Step 12 should run a bounded BAIR baseline comparison: Random-K Student, Uniform-K Student, learned-selector Student, Teacher reference, and optionally an oracle-like top-importance Student. Compare token retention, future MSE, teacher-student ratio, and selected teacher importance.",
    ]
    STEP11F_DOC_PATH.write_text("\n".join(doc) + "\n", encoding="utf-8")


def _failure_payload(error: BaseException, resources: dict[str, Any], elapsed_time_sec: float) -> None:
    payload = {
        "error": repr(error),
        "traceback": traceback.format_exc(),
        "resources": resources,
        "elapsed_time_sec": elapsed_time_sec,
        "oom": "out of memory" in repr(error).lower(),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Student World Model BAIR VideoMAE Smoke Report\n\n"
        "Step 11F failed before producing a complete smoke report.\n\n"
        "```json\n"
        + json.dumps(payload, indent=2)
        + "\n```\n\n"
        "STUDENT_WORLD_MODEL_BAIR_VIDEOMAE_SMOKE_PASS = false\n",
        encoding="utf-8",
    )


def main() -> None:
    start = time.time()
    before = _resource_snapshot()
    config = load_yaml(CONFIG_PATH)
    resources: dict[str, Any] = {"before": before}
    try:
        _ensure_inputs(config)
        _clean_run_dir(config)
        training_summary = run_student_world_model_training(config)
        eval_summary = evaluate_student_world_model(config, training_summary["checkpoint_path"], split="test")
        gap_summary = evaluate_teacher_student_gap(
            config,
            student_checkpoint_path=training_summary["checkpoint_path"],
            split="test",
        )
        inspection = inspect_student_world_model_checkpoint(training_summary["checkpoint_path"])
        after = _resource_snapshot()
        elapsed = time.time() - start
        resources["after"] = after
        resources["elapsed_time_sec"] = elapsed
        resources["max_ram_used_gib"] = max(before.get("ram_used_gib", 0.0), after.get("ram_used_gib", 0.0))
        resources["max_gpu_mem_used_gib"] = max(
            before.get("gpu_mem_used_gib", 0.0),
            after.get("gpu_mem_used_gib", 0.0),
        )
        resources["oom"] = False
        max_ram_gb_warning = float(config.get("resource_limits", {}).get("max_ram_gb_warning", 28.0))
        max_gpu_mem_fraction_warning = float(
            config.get("resource_limits", {}).get("max_gpu_mem_fraction_warning", 0.9)
        )
        gpu_fraction = (
            resources["max_gpu_mem_used_gib"] / after["gpu_mem_total_gib"]
            if after.get("gpu_mem_total_gib", 0.0)
            else 0.0
        )
        resources["gpu_mem_fraction"] = gpu_fraction
        resources["cloud_recommendation"] = (
            "consider 4090 / 48GB for larger-scale BAIR runs"
            if resources["max_ram_used_gib"] >= max_ram_gb_warning or gpu_fraction >= max_gpu_mem_fraction_warning
            else "not required for Step 11F smoke"
        )

        expected_retention = float(config["selection"]["topk"]) / float(training_summary["num_tokens"])
        checks = {
            "summary_exists": Path(training_summary["summary_path"]).exists(),
            "metrics_exists": Path(training_summary["metrics_path"]).exists(),
            "checkpoint_exists": Path(training_summary["checkpoint_path"]).exists(),
            "eval_summary_exists": Path(eval_summary["eval_summary_path"]).exists(),
            "gap_summary_exists": Path(gap_summary["gap_summary_path"]).exists(),
            "num_steps_ok": int(training_summary["num_steps"]) >= 50,
            "initial_loss_finite": _finite(training_summary["initial_loss"]),
            "final_loss_finite": _finite(training_summary["final_loss"]),
            "best_loss_finite": _finite(training_summary["best_loss"]),
            "student_future_mse_finite": _finite(eval_summary["student_future_mse"]),
            "teacher_mse_finite": _finite(gap_summary["teacher_mse"]),
            "student_teacher_ratio_finite": _finite(gap_summary["student_teacher_ratio"]),
            "selector_target_topk_overlap_finite": _finite(eval_summary["selector_target_topk_overlap"]),
            "selected_teacher_importance_mean_finite": _finite(eval_summary["selected_teacher_importance_mean"]),
            "selected_vs_random_importance_gap_finite": _finite(
                eval_summary["selected_vs_random_importance_gap"]
            ),
            "token_retention_ratio_ok": math.isclose(
                float(gap_summary["token_retention_ratio"]),
                expected_retention,
                rel_tol=1e-6,
                abs_tol=1e-9,
            ),
            "selector_frozen": bool(training_summary["selector_frozen"]),
            "ram_below_warning": resources["max_ram_used_gib"] < max_ram_gb_warning,
        }
        pass_flag = all(checks.values())
        training_summary["smoke_checks"] = checks
        Path(training_summary["summary_path"]).write_text(json.dumps(training_summary, indent=2), encoding="utf-8")
        _write_report(
            config,
            training_summary,
            eval_summary,
            gap_summary,
            inspection,
            resources,
            checks,
            pass_flag,
            elapsed,
            oom=False,
        )
        _write_step11f_doc(config, training_summary, eval_summary, gap_summary, resources, pass_flag)
        print(REPORT_PATH.read_text(encoding="utf-8"))
        if not pass_flag:
            raise SystemExit(1)
    except Exception as error:
        elapsed = time.time() - start
        resources["after"] = _resource_snapshot()
        resources["elapsed_time_sec"] = elapsed
        _failure_payload(error, resources, elapsed)
        print(REPORT_PATH.read_text(encoding="utf-8"))
        raise


if __name__ == "__main__":
    main()
