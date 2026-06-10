"""End-to-end Step 6 smoke test for Student selector training."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_student_selector import evaluate_student_selector
from scripts.inspect_student_selector_checkpoint import inspect_student_selector_checkpoint
from scripts.smoke_test_importance_signal_quality import main as run_importance_signal_smoke
from training.student_selector_trainer import run_student_selector_training
from training.train_student_selector import load_yaml


REPORT_PATH = PROJECT_ROOT / "docs" / "STUDENT_SELECTOR_TRAINING_SMOKE_REPORT.md"
STEP6_DOC_PATH = PROJECT_ROOT / "docs" / "STEP6_STUDENT_SELECTOR_TRAINING.md"


def _project_config(config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    updated["data"] = dict(updated["data"])
    updated["output"] = dict(updated["output"])
    updated["data"]["token_shard_dir"] = str(PROJECT_ROOT / "data" / "token_shards" / "structured_toy")
    updated["data"]["importance_shard_dir"] = str(
        PROJECT_ROOT / "data" / "importance_shards" / "structured_toy_teacher"
    )
    updated["output"]["run_root"] = str(PROJECT_ROOT / "runs")
    updated["output"]["run_name"] = "student_selector_structured_toy_v1"
    return updated


def _ensure_structured_importance(config: dict[str, Any]) -> None:
    token_dir = Path(config["data"]["token_shard_dir"])
    importance_dir = Path(config["data"]["importance_shard_dir"])
    if list(token_dir.glob(config["data"].get("token_shard_glob", "tokens_shard_*.pt"))) and list(
        importance_dir.glob(config["data"].get("importance_shard_glob", "importance_shard_*.pt"))
    ):
        return
    run_importance_signal_smoke()


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _write_smoke_report(
    config: dict[str, Any],
    training_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    inspection: dict[str, Any],
    pass_flag: bool,
) -> None:
    command = (
        "/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab "
        "python scripts/smoke_test_student_selector_training.py"
    )
    report = [
        "# Student Selector Training Smoke Report",
        "",
        f"Command: `{command}`",
        "",
        f"- token shard input dir: `{config['data']['token_shard_dir']}`",
        f"- importance shard input dir: `{config['data']['importance_shard_dir']}`",
        f"- run dir: `{training_summary['run_dir']}`",
        f"- checkpoint path: `{training_summary['checkpoint_path']}`",
        f"- num steps: `{training_summary['num_steps']}`",
        f"- initial loss: `{training_summary['initial_loss']}`",
        f"- final loss: `{training_summary['final_loss']}`",
        f"- best loss: `{training_summary['best_loss']}`",
        f"- loss_decreased: `{training_summary['loss_decreased']}`",
        f"- final key score mean: `{training_summary['final_key_score_mean']}`",
        f"- final non-key score mean: `{training_summary['final_non_key_score_mean']}`",
        f"- final key_vs_non_key_gap: `{training_summary['final_key_vs_non_key_gap']}`",
        f"- final top1 hit rate: `{training_summary['final_top1_hit_rate']}`",
        f"- final topk hit rate: `{training_summary['final_topk_hit_rate']}`",
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
        "## Checkpoint Inspection Summary",
        "",
        "```json",
        json.dumps(
            {
                "step": inspection["step"],
                "model_config": inspection["model_config"],
                "parameter_count": inspection["parameter_count"],
                "state_dict_key_count": inspection["state_dict_key_count"],
            },
            indent=2,
        ),
        "```",
        "",
        f"STUDENT_SELECTOR_TRAINING_SMOKE_PASS = {str(pass_flag).lower()}",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_step6_doc(
    training_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    pass_flag: bool,
) -> None:
    doc = [
        "# STEP6 Student Selector Training Report",
        "",
        "## 1. Goal",
        "",
        "Step 6 trains only the Student Attention Selector to predict Step 5.5 `structured_toy` normalized predictive-importance labels. This stage does not connect real V-JEPA / VideoMAE, VLM grounding, or RL.",
        "",
        "## 2. Why Step 6 Is Now Valid",
        "",
        "Step 5.5 produced positive structured importance signal: key token importance is higher than non-key token importance, normalized importance is not all zero, and structured toy top-k recovery reached 1.0.",
        "",
        "## 3. Files Added or Updated",
        "",
        "- `data/student_selector_dataset.py`",
        "- `training/student_selector_trainer.py`",
        "- `training/train_student_selector.py`",
        "- `eval/eval_student_selector.py`",
        "- `scripts/smoke_test_student_selector_training.py`",
        "- `scripts/inspect_student_selector_checkpoint.py`",
        "- `configs/train_student_selector_structured_toy.yaml`",
        "- `tests/test_student_selector_dataset.py`",
        "- `tests/test_student_selector_training_step.py`",
        "- `tests/test_student_selector_eval.py`",
        "",
        "## 4. StudentSelectorDataset",
        "",
        "`StudentSelectorDataset` aligns token shards and importance shards by `sample_id`, returning `past_tokens`, normalized importance labels, raw importance scores, key token masks, task text, and metadata.",
        "",
        "## 5. AttentionSelector",
        "",
        "The selector consumes `[B, N, D]` tokens and emits `[B, N]` logits. Training applies `sigmoid(logits)` and regresses probabilities to `importance_scores_norm`.",
        "",
        "## 6. Training Pipeline",
        "",
        "`structured_toy` token shards plus importance shards feed the paired dataset. The selector is trained with importance regression loss plus a small key/non-key ranking margin loss. Metrics, summary, and checkpoint are written under ignored `runs/`.",
        "",
        "## 7. Smoke Test Result",
        "",
        f"- run dir: `{training_summary['run_dir']}`",
        f"- num steps: `{training_summary['num_steps']}`",
        f"- initial loss: `{training_summary['initial_loss']}`",
        f"- final loss: `{training_summary['final_loss']}`",
        f"- best loss: `{training_summary['best_loss']}`",
        f"- loss_decreased: `{training_summary['loss_decreased']}`",
        f"- key_score_mean: `{training_summary['final_key_score_mean']}`",
        f"- non_key_score_mean: `{training_summary['final_non_key_score_mean']}`",
        f"- key_vs_non_key_gap: `{training_summary['final_key_vs_non_key_gap']}`",
        f"- top1_hit_rate: `{training_summary['final_top1_hit_rate']}`",
        f"- topk_hit_rate: `{training_summary['final_topk_hit_rate']}`",
        f"- STUDENT_SELECTOR_TRAINING_SMOKE_PASS: `{str(pass_flag).lower()}`",
        "",
        "## 8. Eval Result",
        "",
        f"- importance_mse: `{eval_summary['importance_mse']}`",
        f"- key_score_mean: `{eval_summary['key_score_mean']}`",
        f"- non_key_score_mean: `{eval_summary['non_key_score_mean']}`",
        f"- key_vs_non_key_gap: `{eval_summary['key_vs_non_key_gap']}`",
        f"- top1_hit_rate: `{eval_summary['top1_hit_rate']}`",
        f"- topk_hit_rate: `{eval_summary['topk_hit_rate']}`",
        "",
        "## 9. Pytest Result",
        "",
        "Run `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python -m pytest tests -q` after Step 6 changes. The final local summary records the observed result.",
        "",
        "## 10. Git Commit",
        "",
        "Branch: `feature/tgpawb-step3-token-extraction`. Commit hash and push status are recorded after commit/push. No PR is created in this stage.",
        "",
        "## 11. What Was Not Done",
        "",
        "- no real dataset download",
        "- no real V-JEPA / VideoMAE encoder",
        "- no large model download",
        "- no VLM grounding",
        "- no StudentWorldModel full training yet",
        "- no token compressor training yet",
        "- no RL / policy optimization",
        "- no generated `.pt` committed",
        "- no checkpoint committed",
        "- no password or token saved",
        "- no PR created",
        "",
        "## 12. Next Step Recommendation",
        "",
        "Step 7 should connect the selector to the token compressor and StudentWorldModel on `structured_toy`, validating compressed Student future-latent prediction under a low token budget.",
    ]
    STEP6_DOC_PATH.write_text("\n".join(doc) + "\n", encoding="utf-8")


def main() -> None:
    config = _project_config(load_yaml(PROJECT_ROOT / "configs" / "train_student_selector_structured_toy.yaml"))
    _ensure_structured_importance(config)
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    if run_dir.exists():
        shutil.rmtree(run_dir)

    training_summary = run_student_selector_training(config)
    eval_summary = evaluate_student_selector(config, training_summary["checkpoint_path"])
    inspection = inspect_student_selector_checkpoint(training_summary["checkpoint_path"])
    checks = {
        "summary_exists": Path(training_summary["summary_path"]).exists(),
        "metrics_exists": Path(training_summary["metrics_path"]).exists(),
        "checkpoint_exists": Path(training_summary["checkpoint_path"]).exists(),
        "num_steps_ok": int(training_summary["num_steps"]) >= 50,
        "initial_loss_finite": _finite(float(training_summary["initial_loss"])),
        "final_loss_finite": _finite(float(training_summary["final_loss"])),
        "best_loss_finite": _finite(float(training_summary["best_loss"])),
        "loss_decreased": bool(training_summary["loss_decreased"]),
        "key_gt_non_key": training_summary["final_key_score_mean"] > training_summary["final_non_key_score_mean"],
        "top1_hit_rate": training_summary["final_top1_hit_rate"] >= 0.8,
        "topk_hit_rate": training_summary["final_topk_hit_rate"] >= 0.8,
    }
    pass_flag = all(checks.values())
    training_summary["smoke_checks"] = checks
    Path(training_summary["summary_path"]).write_text(json.dumps(training_summary, indent=2), encoding="utf-8")
    _write_smoke_report(config, training_summary, eval_summary, inspection, pass_flag)
    _write_step6_doc(training_summary, eval_summary, pass_flag)
    print(REPORT_PATH.read_text(encoding="utf-8"))
    if not pass_flag:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
