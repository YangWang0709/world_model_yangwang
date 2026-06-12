"""Build or normalize a Step22 user-provided BridgeData V2 manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_user_subset_manifest_tools import (
    build_or_normalize_user_subset_manifest,
    write_manifest_summary,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_user_subset_ingestion_step22.yaml"


def build_manifest_from_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    output = config["output"]
    summary = build_or_normalize_user_subset_manifest(
        dataset["user_subset_root"],
        output["generated_manifest_jsonl"],
        manifest_path=dataset["manifest_path"],
        min_frames=int(dataset["min_frames_per_trajectory"]),
        max_trajectories=int(dataset["max_trajectories_for_validation"]),
    )
    write_manifest_summary(summary, output["manifest_summary_json"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(build_manifest_from_config(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
