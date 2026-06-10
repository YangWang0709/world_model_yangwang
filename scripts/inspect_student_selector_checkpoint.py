"""Inspect a Student selector checkpoint without printing tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.student_selector_trainer import load_student_selector_checkpoint


def inspect_student_selector_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = load_student_selector_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    parameter_count = sum(int(value.numel()) for value in state_dict.values())
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_keys": sorted(checkpoint.keys()),
        "model_config": checkpoint["model_config"],
        "step": int(checkpoint["step"]),
        "metrics_summary": checkpoint.get("metrics_summary", {}),
        "state_dict_key_count": len(state_dict),
        "parameter_count": parameter_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = inspect_student_selector_checkpoint(args.checkpoint)
    print("STUDENT_SELECTOR_CHECKPOINT_INSPECTION_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
