"""Inspect a Student world model checkpoint without printing tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.student_world_model_trainer import load_student_world_model_checkpoint


def inspect_student_world_model_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = load_student_world_model_checkpoint(checkpoint_path, map_location="cpu")
    compressor_state = checkpoint["compressor_state_dict"]
    student_state = checkpoint["student_world_model_state_dict"]
    selector_state = checkpoint.get("selector_state_dict") or {}
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_keys": sorted(checkpoint.keys()),
        "step": int(checkpoint["step"]),
        "selector_checkpoint_path": checkpoint["selector_checkpoint_path"],
        "selector_config": checkpoint["selector_config"],
        "selector_frozen": bool(checkpoint.get("selector_frozen", True)),
        "compressor_config": checkpoint["compressor_config"],
        "student_world_model_config": checkpoint["student_world_model_config"],
        "metrics_summary": checkpoint.get("metrics_summary", {}),
        "selector_parameter_count": sum(int(value.numel()) for value in selector_state.values()),
        "compressor_parameter_count": sum(int(value.numel()) for value in compressor_state.values()),
        "student_world_model_parameter_count": sum(int(value.numel()) for value in student_state.values()),
        "compressor_state_dict_key_count": len(compressor_state),
        "student_world_model_state_dict_key_count": len(student_state),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = inspect_student_world_model_checkpoint(args.checkpoint)
    print("STUDENT_WORLD_MODEL_CHECKPOINT_INSPECTION_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
