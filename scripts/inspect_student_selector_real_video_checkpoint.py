"""Inspect a real VideoMAE Student selector checkpoint without printing tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inspect_student_selector_checkpoint import inspect_student_selector_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = inspect_student_selector_checkpoint(args.checkpoint)
    print("STUDENT_SELECTOR_REAL_VIDEOMAE_CHECKPOINT_INSPECTION_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
