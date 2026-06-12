"""Inventory official BridgeData V2 TFDS shards for Step23."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_tfds_inventory import inventory_bridgedata_v2_tfds, write_inventory_summary

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_mini_shard_step23.yaml"


def inventory_from_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary = inventory_bridgedata_v2_tfds(config)
    write_inventory_summary(summary, config["output"]["inventory_json"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(inventory_from_config(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
