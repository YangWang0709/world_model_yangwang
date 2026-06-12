"""Acquire a safe official BridgeData V2 tiny sample if one exists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_real_tiny_acquisition import acquire_bridgedata_v2_real_tiny_subset
from scripts.probe_bridgedata_v2_download_options import run_probe_from_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_real_tiny_validation_step21.yaml"


def run_acquisition_from_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    probe_path = Path(config["output"]["probe_summary_json"])
    if probe_path.exists():
        probe_summary = json.loads(probe_path.read_text(encoding="utf-8"))
    else:
        probe_summary = run_probe_from_config(config_path)
    return acquire_bridgedata_v2_real_tiny_subset(config, probe_summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = run_acquisition_from_config(args.config)
    print(json.dumps({"download_performed": summary["download_performed"], "safe_stop": summary["safe_stop"]}, indent=2))


if __name__ == "__main__":
    main()
