"""Inspect a Step 4 teacher checkpoint without printing tensor values."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.teacher_trainer import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def inspect_checkpoint(checkpoint_path: str | Path) -> dict:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    parameter_count = int(sum(tensor.numel() for tensor in state_dict.values() if torch.is_tensor(tensor)))
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_keys": sorted(checkpoint.keys()),
        "model_config": checkpoint["model_config"],
        "step": int(checkpoint["step"]),
        "metrics_summary": checkpoint.get("metrics_summary", {}),
        "state_dict_key_count": len(state_dict),
        "parameter_count": parameter_count,
    }
    return summary


def main() -> None:
    args = parse_args()
    print(json.dumps(inspect_checkpoint(args.checkpoint), indent=2))


if __name__ == "__main__":
    main()

