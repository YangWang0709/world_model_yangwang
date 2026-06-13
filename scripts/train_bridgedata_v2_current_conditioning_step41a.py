"""Run Step41A bounded current-conditioning selector diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.bridgedata_v2_current_conditioning_trainer_step41a import train_step41a_current_conditioning

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_current_conditioning_diagnosis_step41a.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    payload = train_step41a_current_conditioning(args.config)
    print(json.dumps(payload["train_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
