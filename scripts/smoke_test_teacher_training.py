"""End-to-end smoke test for Step 4 teacher tiny training."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_teacher_prediction import evaluate_teacher
from scripts.inspect_teacher_checkpoint import inspect_checkpoint
from scripts.smoke_test_token_extraction import main as run_token_extraction_smoke
from training.train_teacher import load_yaml
from training.teacher_trainer import run_teacher_training


REPORT_PATH = PROJECT_ROOT / "docs" / "TEACHER_TRAINING_SMOKE_REPORT.md"


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "train_teacher_dummy.yaml"
    config = load_yaml(config_path)
    shard_dir = Path(config["data"]["token_shard_dir"])
    if not list(shard_dir.glob(config["data"].get("shard_glob", "tokens_shard_*.pt"))):
        run_token_extraction_smoke()

    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    if run_dir.exists():
        shutil.rmtree(run_dir)

    summary = run_teacher_training(config)
    checkpoint_path = Path(summary["checkpoint_path"])
    metrics_path = Path(summary["metrics_path"])
    summary_path = Path(summary["summary_path"])

    checks = {
        "summary_exists": summary_path.exists(),
        "metrics_exists": metrics_path.exists(),
        "checkpoint_exists": checkpoint_path.exists(),
        "num_steps_ok": int(summary["num_steps"]) >= 10,
        "initial_loss_finite": finite(summary["initial_loss"]),
        "final_loss_finite": finite(summary["final_loss"]),
        "best_loss_finite": finite(summary["best_loss"]),
    }
    eval_summary = evaluate_teacher(config, checkpoint_path)
    inspection = inspect_checkpoint(checkpoint_path)
    checks["eval_mse_finite"] = finite(eval_summary["eval_mse"])
    checks["parameter_count_ok"] = int(inspection["parameter_count"]) > 0
    pass_flag = all(checks.values())

    report = [
        "# Teacher Training Smoke Report",
        "",
        "Command: `python scripts/smoke_test_teacher_training.py`",
        "",
        f"- token shard path: `{shard_dir}`",
        f"- run dir: `{run_dir}`",
        f"- checkpoint path: `{checkpoint_path}`",
        f"- metrics path: `{metrics_path}`",
        f"- num steps: `{summary['num_steps']}`",
        f"- initial loss: `{summary['initial_loss']}`",
        f"- final loss: `{summary['final_loss']}`",
        f"- best loss: `{summary['best_loss']}`",
        f"- loss_decreased: `{summary['loss_decreased']}`",
        f"- device: `{summary['device']}`",
        f"- eval mse: `{eval_summary['eval_mse']}`",
        "",
        "## Checkpoint Inspection Summary",
        "",
        "```json",
        json.dumps(
            {
                "step": inspection["step"],
                "parameter_count": inspection["parameter_count"],
                "state_dict_key_count": inspection["state_dict_key_count"],
                "model_config": inspection["model_config"],
            },
            indent=2,
        ),
        "```",
        "",
        f"TEACHER_TRAINING_SMOKE_PASS = {str(pass_flag).lower()}",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    if not pass_flag:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

