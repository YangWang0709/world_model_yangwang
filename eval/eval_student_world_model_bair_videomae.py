"""Evaluate Step 11F Student world model on BAIR VideoMAE token shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_student_world_model import evaluate_student_world_model
from training.train_student_world_model import load_yaml


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_student_world_model_bair_videomae_smoke.yaml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "student_world_model_bair_videomae_smoke_v1"
    / "checkpoints"
    / "student_world_model_step_000500.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--split", default="test", choices=["train", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_student_world_model(load_yaml(args.config), args.checkpoint, split=args.split)
    print("STUDENT_WORLD_MODEL_BAIR_VIDEOMAE_EVAL_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
