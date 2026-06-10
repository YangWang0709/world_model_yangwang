"""End-to-end Step 7 smoke test for compressed Student world model training."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_student_world_model import evaluate_student_world_model
from eval.eval_teacher_student_gap import evaluate_teacher_student_gap
from scripts.inspect_student_world_model_checkpoint import inspect_student_world_model_checkpoint
from scripts.smoke_test_student_selector_training import main as run_student_selector_smoke
from training.student_world_model_trainer import run_student_world_model_training
from training.train_student_world_model import load_yaml


REPORT_PATH = PROJECT_ROOT / "docs" / "STUDENT_WORLD_MODEL_TRAINING_SMOKE_REPORT.md"
STEP7_DOC_PATH = PROJECT_ROOT / "docs" / "STEP7_STUDENT_WORLD_MODEL_TRAINING.md"


def _project_config(config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    updated["data"] = dict(updated["data"])
    updated["selector"] = dict(updated["selector"])
    updated["teacher_reference"] = dict(updated["teacher_reference"])
    updated["output"] = dict(updated["output"])
    updated["data"]["token_shard_dir"] = str(PROJECT_ROOT / "data" / "token_shards" / "structured_toy")
    updated["data"]["importance_shard_dir"] = str(
        PROJECT_ROOT / "data" / "importance_shards" / "structured_toy_teacher"
    )
    updated["selector"]["checkpoint"] = str(
        PROJECT_ROOT
        / "runs"
        / "student_selector_structured_toy_v1"
        / "checkpoints"
        / "student_selector_step_000200.pt"
    )
    updated["teacher_reference"]["checkpoint"] = str(
        PROJECT_ROOT
        / "runs"
        / "teacher_structured_toy_tiny_v1"
        / "checkpoints"
        / "teacher_world_model_step_000100.pt"
    )
    updated["output"]["run_root"] = str(PROJECT_ROOT / "runs")
    updated["output"]["run_name"] = "student_world_model_structured_toy_v1"
    return updated


def _ensure_step7_inputs(config: dict[str, Any]) -> None:
    token_dir = Path(config["data"]["token_shard_dir"])
    importance_dir = Path(config["data"]["importance_shard_dir"])
    selector_checkpoint = Path(config["selector"]["checkpoint"])
    teacher_checkpoint = Path(config["teacher_reference"]["checkpoint"])
    token_ready = bool(list(token_dir.glob(config["data"].get("token_shard_glob", "tokens_shard_*.pt"))))
    importance_ready = bool(
        list(importance_dir.glob(config["data"].get("importance_shard_glob", "importance_shard_*.pt")))
    )
    if token_ready and importance_ready and selector_checkpoint.exists() and teacher_checkpoint.exists():
        return
    run_student_selector_smoke()


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _write_smoke_report(
    config: dict[str, Any],
    training_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    gap_summary: dict[str, Any],
    inspection: dict[str, Any],
    pass_flag: bool,
) -> None:
    command = (
        "/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab "
        "python scripts/smoke_test_student_world_model_training.py"
    )
    report = [
        "# Student World Model Training Smoke Report",
        "",
        f"Command: `{command}`",
        "",
        f"- token shard input dir: `{config['data']['token_shard_dir']}`",
        f"- importance shard input dir: `{config['data']['importance_shard_dir']}`",
        f"- selector checkpoint: `{training_summary['selector_checkpoint_path']}`",
        f"- teacher checkpoint: `{gap_summary['teacher_checkpoint_path']}`",
        f"- run dir: `{training_summary['run_dir']}`",
        f"- checkpoint path: `{training_summary['checkpoint_path']}`",
        f"- num steps: `{training_summary['num_steps']}`",
        f"- initial loss: `{training_summary['initial_loss']}`",
        f"- final loss: `{training_summary['final_loss']}`",
        f"- best loss: `{training_summary['best_loss']}`",
        f"- loss_decreased: `{training_summary['loss_decreased']}`",
        f"- student future MSE: `{eval_summary['student_future_mse']}`",
        f"- teacher future MSE: `{gap_summary['teacher_future_mse']}`",
        f"- student_teacher_gap: `{gap_summary['student_teacher_gap']}`",
        f"- student_teacher_ratio: `{gap_summary['student_teacher_ratio']}`",
        f"- selected top1 hit rate: `{eval_summary['top1_hit_rate']}`",
        f"- selected topk hit rate: `{eval_summary['topk_hit_rate']}`",
        f"- selected key coverage: `{eval_summary['selected_key_coverage']}`",
        f"- token retention ratio: `{gap_summary['token_retention_ratio']}`",
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
        "## Teacher/Student Gap Summary",
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
                "step": inspection["step"],
                "selector_frozen": inspection["selector_frozen"],
                "compressor_config": inspection["compressor_config"],
                "student_world_model_config": inspection["student_world_model_config"],
                "compressor_parameter_count": inspection["compressor_parameter_count"],
                "student_world_model_parameter_count": inspection["student_world_model_parameter_count"],
            },
            indent=2,
        ),
        "```",
        "",
        f"STUDENT_WORLD_MODEL_TRAINING_SMOKE_PASS = {str(pass_flag).lower()}",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_step7_doc(
    training_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    gap_summary: dict[str, Any],
    pass_flag: bool,
) -> None:
    doc = [
        "# STEP7 Student World Model Training Report",
        "",
        "## 1. Goal",
        "",
        "Step 7 trains a compressed Student world-model chain on `structured_toy`: `past_tokens -> frozen Step6 AttentionSelector -> topK selected tokens -> TokenCompressor -> StudentWorldModel -> predicted future latent`.",
        "",
        "## 2. Default Training Scope",
        "",
        "The Step6 selector is frozen by default. Only `TokenCompressor` and `StudentWorldModel` are optimized unless the config explicitly sets `selector.frozen: false`.",
        "",
        "## 3. Target",
        "",
        "`future_tokens` are converted to a `[B, D]` target with the same helper used by teacher training. Rank-3 future tokens are mean-pooled over the token dimension; rank-2 targets are used directly.",
        "",
        "## 4. Files Added or Updated",
        "",
        "- `configs/train_student_world_model_structured_toy.yaml`",
        "- `models/token_compressor.py`",
        "- `models/student_world_model.py`",
        "- `training/student_world_model_trainer.py`",
        "- `training/train_student_world_model.py`",
        "- `eval/eval_student_world_model.py`",
        "- `eval/eval_teacher_student_gap.py`",
        "- `scripts/smoke_test_student_world_model_training.py`",
        "- `scripts/inspect_student_world_model_checkpoint.py`",
        "- `tests/test_student_world_model_training_step.py`",
        "- `tests/test_student_world_model_eval.py`",
        "- `tests/test_teacher_student_gap.py`",
        "",
        "## 5. Smoke Test Result",
        "",
        f"- run dir: `{training_summary['run_dir']}`",
        f"- checkpoint: `{training_summary['checkpoint_path']}`",
        f"- selector checkpoint: `{training_summary['selector_checkpoint_path']}`",
        f"- selector_frozen: `{training_summary['selector_frozen']}`",
        f"- num steps: `{training_summary['num_steps']}`",
        f"- initial loss: `{training_summary['initial_loss']}`",
        f"- final loss: `{training_summary['final_loss']}`",
        f"- best loss: `{training_summary['best_loss']}`",
        f"- loss_decreased: `{training_summary['loss_decreased']}`",
        f"- selected_top1_hit_rate: `{eval_summary['top1_hit_rate']}`",
        f"- selected_topk_hit_rate: `{eval_summary['topk_hit_rate']}`",
        f"- selected_key_coverage: `{eval_summary['selected_key_coverage']}`",
        f"- token_retention_ratio: `{gap_summary['token_retention_ratio']}`",
        f"- STUDENT_WORLD_MODEL_TRAINING_SMOKE_PASS: `{str(pass_flag).lower()}`",
        "",
        "## 6. Teacher/Student Future Prediction Gap",
        "",
        f"- teacher_future_mse: `{gap_summary['teacher_future_mse']}`",
        f"- student_future_mse: `{gap_summary['student_future_mse']}`",
        f"- student_teacher_gap: `{gap_summary['student_teacher_gap']}`",
        f"- student_teacher_ratio: `{gap_summary['student_teacher_ratio']}`",
        "",
        "## 7. Pytest Result",
        "",
        "Run `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python -m pytest tests -q` after Step 7 changes. The final local summary records the observed result.",
        "",
        "## 8. Git Commit",
        "",
        "Branch: `feature/tgpawb-step3-token-extraction`. Commit hash and push status are recorded after commit/push. No PR is created in this stage.",
        "",
        "## 9. What Was Not Done",
        "",
        "- no real dataset download",
        "- no real V-JEPA / VideoMAE encoder",
        "- no large model download",
        "- no VLM grounding",
        "- no RL / policy optimization",
        "- no generated `.pt` committed",
        "- no checkpoint committed",
        "- no password or token saved",
        "- no PR created",
        "",
        "## 10. Next Step Recommendation",
        "",
        "Move from the controlled `structured_toy` setup to a slightly less synthetic offline token set only after the compressed Student chain keeps stable coverage and finite teacher/student gap metrics.",
    ]
    STEP7_DOC_PATH.write_text("\n".join(doc) + "\n", encoding="utf-8")


def main() -> None:
    config = _project_config(load_yaml(PROJECT_ROOT / "configs" / "train_student_world_model_structured_toy.yaml"))
    _ensure_step7_inputs(config)
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    if run_dir.exists():
        shutil.rmtree(run_dir)

    training_summary = run_student_world_model_training(config)
    eval_summary = evaluate_student_world_model(config, training_summary["checkpoint_path"])
    gap_summary = evaluate_teacher_student_gap(config, training_summary["checkpoint_path"])
    inspection = inspect_student_world_model_checkpoint(training_summary["checkpoint_path"])
    checks = {
        "summary_exists": Path(training_summary["summary_path"]).exists(),
        "metrics_exists": Path(training_summary["metrics_path"]).exists(),
        "checkpoint_exists": Path(training_summary["checkpoint_path"]).exists(),
        "num_steps_ok": int(training_summary["num_steps"]) >= 100,
        "initial_loss_finite": _finite(training_summary["initial_loss"]),
        "final_loss_finite": _finite(training_summary["final_loss"]),
        "best_loss_finite": _finite(training_summary["best_loss"]),
        "loss_decreased": bool(training_summary["loss_decreased"]),
        "student_future_mse_finite": _finite(eval_summary["student_future_mse"]),
        "teacher_future_mse_finite": _finite(gap_summary["teacher_future_mse"]),
        "student_teacher_ratio_finite": _finite(gap_summary["student_teacher_ratio"]),
        "selected_top1_hit_rate": eval_summary["top1_hit_rate"] >= 0.8,
        "selected_topk_hit_rate": eval_summary["topk_hit_rate"] >= 0.8,
        "token_retention_ratio": 0.0 < gap_summary["token_retention_ratio"] < 1.0,
    }
    pass_flag = all(checks.values())
    training_summary["smoke_checks"] = checks
    Path(training_summary["summary_path"]).write_text(json.dumps(training_summary, indent=2), encoding="utf-8")
    _write_smoke_report(config, training_summary, eval_summary, gap_summary, inspection, pass_flag)
    _write_step7_doc(training_summary, eval_summary, gap_summary, pass_flag)
    print(REPORT_PATH.read_text(encoding="utf-8"))
    if not pass_flag:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
