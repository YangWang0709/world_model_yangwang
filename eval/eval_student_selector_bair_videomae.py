"""Evaluate a Student selector on BAIR VideoMAE importance labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_student_selector import evaluate_student_selector
from training.train_student_selector import load_yaml


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_student_selector_bair_videomae_smoke.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_student_selector(load_yaml(args.config), args.checkpoint, split=args.split)
    print("STUDENT_SELECTOR_BAIR_VIDEOMAE_EVAL_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
