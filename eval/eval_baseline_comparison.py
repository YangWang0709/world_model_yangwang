"""Regenerate Step 8 baseline comparison summaries from sub-run summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.baseline_student_world_model_trainer import write_baseline_summaries


def load_baseline_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    run_path = Path(run_dir)
    rows = []
    for summary_path in sorted(run_path.glob("*_seed*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if not isinstance(summary, dict):
            raise ValueError(f"Summary must be a JSON object: {summary_path}")
        rows.append(summary)
    if not rows:
        raise FileNotFoundError(f"No baseline sub-run summary files found under {run_path}")
    return rows


def evaluate_baseline_comparison(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows = load_baseline_rows(run_path)
    result = write_baseline_summaries(rows, run_dir=run_path)
    payload = {
        "run_dir": str(run_path),
        "num_rows": len(rows),
        "rows": rows,
        **result,
    }
    print(json.dumps(payload, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/baseline_comparison_structured_toy_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    summary = evaluate_baseline_comparison(run_dir)
    print("BASELINE_COMPARISON_EVAL_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
