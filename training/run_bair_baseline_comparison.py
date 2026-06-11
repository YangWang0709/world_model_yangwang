"""CLI entrypoint for Step 12 BAIR baseline comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.baseline_student_world_model_trainer import train_baseline_comparison
from training.run_baseline_comparison import load_yaml


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline_comparison_bair_videomae_smoke.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def run_bair_baseline_comparison(config: dict[str, Any]) -> dict[str, Any]:
    return train_baseline_comparison(config)


def main() -> None:
    args = parse_args()
    summary = run_bair_baseline_comparison(load_yaml(args.config))
    print("BAIR_BASELINE_COMPARISON_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
