"""Evaluate Step23.5 BridgeData V2 TFDS/RLDS field resolution outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_rlds_field_resolver import is_metadata_image_flag

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_field_resolver_step23_5.yaml"


def evaluate_field_resolver(config_path: Path = DEFAULT_CONFIG, docs_dir: Path | None = None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = config["output"]
    resolved = _read_json(output["resolved_fields_json"])
    manifest = _read_json(output["resolved_manifest_summary_json"])
    windows = _read_json(output["resolved_window_summary_json"])
    image_field = resolved.get("image_field")
    rejected_flags = [
        item["field"]
        for item in resolved.get("image_rejected_fields", [])
        if "episode_metadata/has_image_" in item.get("field", "")
    ]
    pass_gate = bool(
        resolved.get("image_field_valid")
        and isinstance(image_field, str)
        and image_field.startswith("steps/observation/image_")
        and not is_metadata_image_flag(image_field)
        and resolved.get("language_field") != "steps/language_embedding"
        and resolved.get("action_field") == "steps/action"
        and manifest.get("num_manifest_records", 0) > 0
        and windows.get("num_windows", 0) > 0
        and not resolved.get("download_performed", False)
        and not windows.get("training_performed", False)
        and not windows.get("token_extraction_performed", False)
        and not windows.get("importance_generation_performed", False)
    )
    summary = {
        "stage": "bridgedata_v2_tfds_field_resolver_step23_5",
        "pass": pass_gate,
        "safe_stop": not pass_gate,
        "image_field_resolved": image_field,
        "image_field_valid": bool(resolved.get("image_field_valid")),
        "image_field_is_metadata_flag": is_metadata_image_flag(image_field),
        "rejected_metadata_image_flags": rejected_flags,
        "action_field": resolved.get("action_field"),
        "action_used_as_input": False,
        "language_field": resolved.get("language_field"),
        "language_used_as_input": False,
        "goal_field": resolved.get("goal_field"),
        "goal_used_as_input": False,
        "num_manifest_records": int(manifest.get("num_manifest_records") or 0),
        "num_windows": int(windows.get("num_windows") or 0),
        "resolved_manifest_path": output["resolved_manifest_jsonl"],
        "resolved_window_manifest_path": output["resolved_window_manifest_jsonl"],
        "download_performed": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "recommended_step24": {
            "name": "BridgeData V2 TFDS mini-shard token extraction dry-run",
            "condition": "resolved image field is steps/observation/image_0 and manifest/window regenerated",
            "no_training": True,
        },
        "resolved_fields": resolved,
        "manifest_summary": manifest,
        "window_summary": windows,
    }
    rendered = render_field_resolver_report(summary)
    Path(output["eval_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(output["eval_json"]).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    Path(output["eval_md"]).write_text(rendered, encoding="utf-8")
    docs_root = docs_dir if docs_dir is not None else PROJECT_ROOT / "docs"
    docs_root.mkdir(parents=True, exist_ok=True)
    (docs_root / "BRIDGEDATA_V2_TFDS_FIELD_RESOLVER_REPORT.md").write_text(rendered, encoding="utf-8")
    return summary


def render_field_resolver_report(summary: dict[str, Any]) -> str:
    status = "Field resolver patch passed." if summary["pass"] else "Field resolver patch did not pass."
    return "\n".join(
        [
            "# BridgeData V2 TFDS Field Resolver Report",
            "",
            status,
            "",
            "## Resolved Fields",
            "",
            f"- image_field: `{summary['image_field_resolved']}`",
            f"- image_field_valid: `{str(summary['image_field_valid']).lower()}`",
            f"- image_field_is_metadata_flag: `{str(summary['image_field_is_metadata_flag']).lower()}`",
            f"- rejected_metadata_image_flags: `{summary['rejected_metadata_image_flags']}`",
            f"- action_field: `{summary['action_field']}`",
            f"- action_used_as_input: `{str(summary['action_used_as_input']).lower()}`",
            f"- language_field: `{summary['language_field']}`",
            f"- language_used_as_input: `{str(summary['language_used_as_input']).lower()}`",
            f"- goal_field: `{summary['goal_field']}`",
            f"- goal_used_as_input: `{str(summary['goal_used_as_input']).lower()}`",
            "",
            "## Regenerated Outputs",
            "",
            f"- num_manifest_records: `{summary['num_manifest_records']}`",
            f"- num_windows: `{summary['num_windows']}`",
            f"- resolved_manifest_path: `{summary['resolved_manifest_path']}`",
            f"- resolved_window_manifest_path: `{summary['resolved_window_manifest_path']}`",
            "",
            "## Boundaries",
            "",
            "- no download",
            "- no training",
            "- no token extraction",
            "- no importance generation",
            "- action/language/goal remain metadata only",
            "",
            "## Recommended Step24",
            "",
            f"- name: `{summary['recommended_step24']['name']}`",
            f"- condition: `{summary['recommended_step24']['condition']}`",
            f"- no_training: `{str(summary['recommended_step24']['no_training']).lower()}`",
            "",
        ]
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(evaluate_field_resolver(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
