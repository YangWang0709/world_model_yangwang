"""Inspect Step17 context bottleneck outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bair_context_bottleneck_validation import load_yaml, write_context_bottleneck_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "context_bottleneck_bair_1000_128.yaml"))
    return parser.parse_args()


def main() -> None:
    config = load_yaml(parse_args().config)
    summary_path = Path(config["output"]["summary_json"])
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = write_context_bottleneck_summaries(config=config)
    compact = {
        "pass": summary.get("sanity_gate_pass"),
        "current_only_mse": summary.get("current_only_mse"),
        "learned_context_mse": summary.get("learned_context_mse"),
        "random_context_mse": summary.get("random_context_mse"),
        "uniform_context_mse": summary.get("uniform_context_mse"),
        "teacher_context_importance_topk_mse": summary.get("teacher_context_importance_topk_mse"),
        "context_gain_over_current_only": summary.get("context_gain_over_current_only"),
        "cloud_required": summary.get("cloud_required"),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
