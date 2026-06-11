"""Regenerate Step 13 selector-level ablation summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.selector_ablation_trainer import (
    load_selector_ablation_rows,
    write_selector_ablation_summaries,
)


DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "selector_ablation_bair_videomae_smoke_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    rows = load_selector_ablation_rows(run_dir)
    summary = write_selector_ablation_summaries(rows, run_dir=run_dir)
    print("BAIR_SELECTOR_ABLATION_EVAL_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
