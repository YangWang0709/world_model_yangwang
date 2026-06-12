"""Inspect a Step22 user-provided BridgeData V2 subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_user_subset_ingestion import (
    inspect_user_subset_directory,
    write_user_subset_inspection_outputs,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_user_subset_ingestion_step22.yaml"


def inspect_from_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    output = config["output"]
    summary = inspect_user_subset_directory(
        dataset["user_subset_root"],
        manifest_path=dataset["manifest_path"],
        max_trajectories=int(dataset["max_trajectories_for_validation"]),
    )
    write_user_subset_inspection_outputs(summary, output["subset_inspection_json"], output["subset_inspection_md"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(inspect_from_config(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
