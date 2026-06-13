"""Run Step36 bounded proxy patch/token selector smoke training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.bridgedata_v2_proxy_patch_selector_trainer import train_step36_proxy_patch_selector

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_proxy_patch_selector_train_step36.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    payload = train_step36_proxy_patch_selector(args.config)
    print(json.dumps(payload["gate_decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
