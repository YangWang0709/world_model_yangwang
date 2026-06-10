"""CLI entrypoint for Step 6 Student selector training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.student_selector_trainer import run_student_selector_training


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_student_selector_structured_toy.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_student_selector_training(load_yaml(args.config))
    print("STUDENT_SELECTOR_TRAINING_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
