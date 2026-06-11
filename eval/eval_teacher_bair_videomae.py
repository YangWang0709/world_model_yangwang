"""Evaluate Step 11C TeacherWorldModel on BAIR VideoMAE test token shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_teacher_prediction import evaluate_teacher, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_teacher_bair_videomae_smoke.yaml")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_teacher(load_yaml(args.config), args.checkpoint)
    print("TEACHER_BAIR_VIDEOMAE_EVAL_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
