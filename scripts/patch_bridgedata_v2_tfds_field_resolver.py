"""Resolve Step23 TFDS/RLDS fields without downloading or reading shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_rlds_field_resolver import resolve_rlds_fields

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_field_resolver_step23_5.yaml"


def resolve_from_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    schema_path = Path(config["input"]["step23_schema_summary"])
    output_path = Path(config["output"]["resolved_fields_json"])
    if not schema_path.exists():
        summary = {
            "stage": "bridgedata_v2_tfds_field_resolver_step23_5",
            "safe_stop": True,
            "reason": "Step23 schema summary is missing.",
            "download_performed": False,
            "training_performed": False,
            "token_extraction_performed": False,
            "importance_generation_performed": False,
        }
    else:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        resolved = resolve_rlds_fields(schema.get("candidate_fields", {}), config.get("field_policy"))
        summary = {
            "stage": "bridgedata_v2_tfds_field_resolver_step23_5",
            "safe_stop": False,
            "reason": None,
            **resolved,
            "download_performed": False,
            "training_performed": False,
            "token_extraction_performed": False,
            "importance_generation_performed": False,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(resolve_from_config(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
