"""Regenerate Step 13 downstream and combined ablation summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.selector_ablation_trainer import (
    load_downstream_ablation_rows,
    load_selector_ablation_rows,
    write_downstream_ablation_summaries,
    write_selector_ablation_summaries,
)


DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "selector_ablation_bair_videomae_smoke_v1"
DEFAULT_STEP12_SUMMARY = PROJECT_ROOT / "runs" / "baseline_comparison_bair_videomae_smoke_v1" / "baseline_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--step12-summary", default=str(DEFAULT_STEP12_SUMMARY))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    step12_summary = Path(args.step12_summary)
    if not step12_summary.is_absolute():
        step12_summary = PROJECT_ROOT / step12_summary
    selector_rows = load_selector_ablation_rows(run_dir)
    selector_summary = write_selector_ablation_summaries(selector_rows, run_dir=run_dir)
    downstream_rows = load_downstream_ablation_rows(run_dir)
    summary = write_downstream_ablation_summaries(
        downstream_rows,
        run_dir=run_dir,
        step12_summary_json=step12_summary,
        selector_summary={
            "rows": selector_rows,
            "aggregate": selector_summary["selector_aggregate"],
        },
    )
    print("BAIR_SELECTOR_DOWNSTREAM_ABLATION_EVAL_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
