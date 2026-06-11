"""Export a small BAIR Robot Pushing subset into .pt clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bair_subset_export import export_bair_subset


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bair_robot_pushing_subset.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_bair_subset(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"BAIR_SUBSET_EXPORT_PASS = {str(summary.get('export_success', False)).lower()}")


if __name__ == "__main__":
    main()
