"""Generate Step30B trained-teacher occlusion labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_tfds_occlusion_teacher_labels import generate_step30b_occlusion_teacher_labels

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_occlusion_teacher_step30b.yaml"


def generate_step30b_teacher_labels(config_path: str | Path = DEFAULT_CONFIG, training_bundle: dict | None = None) -> dict:
    return generate_step30b_occlusion_teacher_labels(config_path, training_bundle=training_bundle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    payload = generate_step30b_teacher_labels(args.config)
    print(json.dumps(payload["teacher_importance_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
