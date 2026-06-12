"""Train Step30B small predictor teacher from existing Step30A tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.bridgedata_v2_tfds_occlusion_teacher_trainer import train_step30b_teachers_from_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_occlusion_teacher_step30b.yaml"


def train_step30b_occlusion_teacher(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    return train_step30b_teachers_from_config(config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    payload = train_step30b_occlusion_teacher(args.config)
    print(json.dumps(payload["teacher_train_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
