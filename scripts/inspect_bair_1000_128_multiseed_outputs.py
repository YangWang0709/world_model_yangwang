"""Print a compact digest for Step 15 BAIR 1000/128 multi-seed validation outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bair_1000_128_multiseed_validation import DEFAULT_CONFIG, load_yaml, write_multiseed_validation_summary


def main() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    summary_path = Path(config["output"]["summary_json"])
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = write_multiseed_validation_summary(config)
    digest = {
        "stage": summary.get("stage"),
        "train_samples": summary.get("train_samples"),
        "test_samples": summary.get("test_samples"),
        "token_shape": summary.get("token_shape"),
        "teacher_eval_mse": summary.get("teacher_eval_mse"),
        "baseline_learned_mse_mean": summary.get("baseline_learned_mse_mean"),
        "baseline_learned_mse_std": summary.get("baseline_learned_mse_std"),
        "baseline_random_mse_mean": summary.get("baseline_random_mse_mean"),
        "baseline_random_mse_std": summary.get("baseline_random_mse_std"),
        "baseline_uniform_mse": summary.get("baseline_uniform_mse"),
        "baseline_teacher_importance_topk_mse": summary.get("baseline_teacher_importance_topk_mse"),
        "sanity_gate_pass": summary.get("sanity_gate_pass"),
    }
    print(json.dumps(digest, indent=2))


if __name__ == "__main__":
    main()
