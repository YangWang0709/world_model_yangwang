"""Inspect BridgeData V2 RLDS schema through the isolated TFDS environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_rlds_schema_inspector import (
    inspect_tfds_dataset_root,
    safe_stop_schema_summary,
    write_schema_summary,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_mini_shard_step23.yaml"


def inspect_from_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tfds_python = Path(config["paths"]["tfds_env_python"])
    output_path = Path(config["output"]["rlds_schema_json"])
    if not tfds_python.exists() or not os.access(tfds_python, os.X_OK):
        summary = safe_stop_schema_summary(False, Path(config["paths"]["tfds_dataset_root"]).exists(), "TFDS env python is missing.")
        write_schema_summary(summary, output_path)
        return summary
    result = subprocess.run(
        [str(tfds_python), str(Path(__file__).resolve()), "--worker", "--config", str(config_path)],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        summary = safe_stop_schema_summary(
            False,
            Path(config["paths"]["tfds_dataset_root"]).exists(),
            "TFDS worker failed: " + "\n".join(((result.stdout or "") + "\n" + (result.stderr or "")).splitlines()[-20:]),
        )
        write_schema_summary(summary, output_path)
        return summary
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    summary = safe_stop_schema_summary(False, Path(config["paths"]["tfds_dataset_root"]).exists(), "TFDS worker did not write schema summary.")
    write_schema_summary(summary, output_path)
    return summary


def worker(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    summary = inspect_tfds_dataset_root(
        config["paths"]["tfds_dataset_root"],
        max_episodes_to_scan=int(config["sample_limits"]["max_episodes_to_scan"]),
        min_trajectory_len=int(config["window"]["min_trajectory_len"]),
    )
    write_schema_summary(summary, config["output"]["rlds_schema_json"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    summary = worker(args.config) if args.worker else inspect_from_config(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
