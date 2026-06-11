"""Print a compact digest for Step 14 BAIR 500/64 scale validation outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bair_500_64_scale_validation import DEFAULT_CONFIG, load_yaml, write_scale_validation_summary


def main() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    summary_path = Path(config["output"]["summary_json"])
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = write_scale_validation_summary(config)
    digest = {
        "stage": summary.get("stage"),
        "train_samples": summary.get("train_samples"),
        "test_samples": summary.get("test_samples"),
        "token_shape": summary.get("token_shape"),
        "teacher_eval_mse": summary.get("teacher_eval_mse"),
        "student_future_mse": summary.get("student_future_mse"),
        "student_teacher_ratio": summary.get("student_teacher_ratio"),
        "baseline_learned_mse": summary.get("baseline_learned_mse"),
        "baseline_random_mean_mse": summary.get("baseline_random_mean_mse"),
        "baseline_uniform_mse": summary.get("baseline_uniform_mse"),
        "baseline_teacher_importance_topk_mse": summary.get("baseline_teacher_importance_topk_mse"),
        "sanity_gate_pass": summary.get("sanity_gate_pass"),
    }
    print(json.dumps(digest, indent=2))


if __name__ == "__main__":
    main()
