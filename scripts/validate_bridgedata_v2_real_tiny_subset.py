"""Validate a real BridgeData V2 tiny subset and reuse the Step20 window builder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_real_format_validator import validate_bridgedata_v2_real_tiny_subset
from scripts.acquire_bridgedata_v2_real_tiny_subset import run_acquisition_from_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_real_tiny_validation_step21.yaml"


def run_validation_from_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    acquisition_path = Path(config["output"]["acquisition_summary_json"])
    if acquisition_path.exists():
        acquisition_summary = json.loads(acquisition_path.read_text(encoding="utf-8"))
    else:
        acquisition_summary = run_acquisition_from_config(config_path)
    return validate_bridgedata_v2_real_tiny_subset(config, acquisition_summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = run_validation_from_config(args.config)
    print(
        json.dumps(
            {
                "real_format_validated": summary["real_format_validated"],
                "safe_stop": summary["safe_stop"],
                "num_windows": summary["num_windows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
