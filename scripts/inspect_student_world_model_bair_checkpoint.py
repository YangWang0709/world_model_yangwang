"""Inspect a Step 11F BAIR Student world model checkpoint without printing tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inspect_student_world_model_checkpoint import inspect_student_world_model_checkpoint


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "student_world_model_bair_videomae_smoke_v1"
    / "checkpoints"
    / "student_world_model_step_000500.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = inspect_student_world_model_checkpoint(args.checkpoint)
    print("STUDENT_WORLD_MODEL_BAIR_CHECKPOINT_INSPECTION_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
