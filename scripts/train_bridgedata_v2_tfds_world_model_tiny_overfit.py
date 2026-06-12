"""Run Step27 BridgeData TFDS tiny world-model overfit training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.bridgedata_v2_tfds_tiny_overfit_trainer import run_tiny_overfit_training

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_world_model_tiny_overfit_step27.yaml"


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = run_tiny_overfit_training(args.config)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
