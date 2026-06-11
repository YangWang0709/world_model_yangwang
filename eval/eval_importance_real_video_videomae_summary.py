"""Evaluate Step 10B real VideoMAE predictive importance shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_importance_summary import evaluate_importance_dir


DEFAULT_IMPORTANCE_DIR = (
    PROJECT_ROOT / "data" / "importance_shards" / "real_video_videomae_teacher_smoke"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance-dir", default=str(DEFAULT_IMPORTANCE_DIR))
    parser.add_argument("--output-path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_path or str(Path(args.importance_dir) / "eval_importance_summary.json")
    evaluate_importance_dir(args.importance_dir, output_path=output_path)


if __name__ == "__main__":
    main()
