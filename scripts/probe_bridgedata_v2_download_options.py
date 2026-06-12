"""Probe official BridgeData V2 download options without downloading data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_download_probe import probe_bridgedata_official_download_options, write_probe_summary

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_real_tiny_validation_step21.yaml"


def run_probe_from_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary = probe_bridgedata_official_download_options(config)
    write_probe_summary(config["output"]["probe_summary_json"], summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = run_probe_from_config(args.config)
    print(json.dumps({"safe_candidate_found": summary["safe_candidate_found"], "safe_stop": summary["safe_stop"]}, indent=2))


if __name__ == "__main__":
    main()
