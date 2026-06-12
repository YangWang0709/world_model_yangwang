"""Build a BridgeData manifest from inspected TFDS/RLDS mini episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_rlds_to_manifest import schema_summary_to_manifest_records, write_manifest_and_summary

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_mini_shard_step23.yaml"


def build_manifest_from_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    schema_path = Path(config["output"]["rlds_schema_json"])
    if not schema_path.exists():
        schema = {"safe_stop": True, "reason": "RLDS schema summary is missing.", "episode_summaries": []}
    else:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    records = []
    if not schema.get("safe_stop"):
        records = schema_summary_to_manifest_records(
            schema,
            min_frames=int(config["window"]["min_trajectory_len"]),
            max_valid_trajectories=int(config["sample_limits"]["max_valid_trajectories"]),
            field_policy=config.get("field_policy"),
        )
    summary = write_manifest_and_summary(
        records,
        config["output"]["tfds_manifest_jsonl"],
        config["output"]["tfds_manifest_summary_json"],
        schema_summary=schema,
        min_valid_trajectories=int(config["sample_limits"]["min_valid_trajectories"]),
    )
    if schema.get("safe_stop") and summary["safe_stop"]:
        summary["reason"] = schema.get("reason") or summary["reason"]
        Path(config["output"]["tfds_manifest_summary_json"]).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(build_manifest_from_config(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
