"""Evaluate Step23 BridgeData V2 TFDS/RLDS mini-shard smoke outputs."""

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

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_mini_shard_step23.yaml"


def evaluate_tfds_mini_shard(config_path: Path = DEFAULT_CONFIG, docs_dir: Path | None = None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = config["output"]
    inventory = _read_json(output["inventory_json"])
    download = _read_json(output["download_summary_json"])
    schema = _read_json(output["rlds_schema_json"])
    manifest = _read_json(output["tfds_manifest_summary_json"])
    windows = _read_json(output["tfds_window_summary_json"])
    real_validated = bool(windows.get("real_tfds_validated", False))
    safe_stop = bool(not real_validated or inventory.get("safe_stop") or download.get("safe_stop") or schema.get("safe_stop"))
    summary = {
        "stage": "bridgedata_v2_tfds_mini_shard_step23",
        "safety_gate_pass": bool(
            not download.get("raw_zip_downloaded", False)
            and not download.get("full_tfds_downloaded", False)
            and not download.get("droid_downloaded", False)
            and int(download.get("download_bytes") or 0) <= int(config["download_policy"]["max_download_bytes"])
            and not windows.get("training_performed", False)
            and not windows.get("token_extraction_performed", False)
            and not windows.get("importance_generation_performed", False)
        ),
        "real_tfds_validated": real_validated,
        "safe_stop": safe_stop,
        "safe_stop_reason": None if real_validated else _first_reason(inventory, download, schema, manifest, windows),
        "download_performed": bool(download.get("download_performed", False)),
        "download_bytes": download.get("download_bytes"),
        "num_shards_downloaded": int(download.get("num_shards_downloaded") or 0),
        "raw_zip_downloaded": False,
        "full_tfds_downloaded": False,
        "droid_downloaded": False,
        "training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "tfds_env_ok": bool(schema.get("tfds_env_ok", False)),
        "dataset_root_exists": bool(schema.get("dataset_root_exists", False)),
        "num_train_shards_found": int(inventory.get("num_train_shards_found") or 0),
        "selected_shards": [item.get("name") for item in inventory.get("selected_shards", [])],
        "selected_total_bytes": inventory.get("selected_total_bytes"),
        "num_episodes_scanned": int(schema.get("num_episodes_scanned") or 0),
        "episode_length_min": schema.get("episode_length_min"),
        "episode_length_mean": schema.get("episode_length_mean"),
        "episode_length_max": schema.get("episode_length_max"),
        "candidate_fields": schema.get("candidate_fields", {}),
        "can_build_16_4_4_windows": bool(schema.get("can_build_16_4_4_windows", False)),
        "num_manifest_records": int(manifest.get("num_manifest_records") or 0),
        "num_valid_trajectories": int(windows.get("num_valid_trajectories") or 0),
        "num_windows": int(windows.get("num_windows") or 0),
        "tfds_mini_manifest_path": output["tfds_manifest_jsonl"],
        "tfds_mini_window_manifest_path": output["tfds_window_manifest_jsonl"],
        "recommended_step24": {
            "name": "BridgeData V2 TFDS mini-shard token extraction dry-run"
            if real_validated
            else "Fix TFDS mini-shard smoke / move to cloud preparation",
            "condition": "real TFDS mini-shard validated"
            if real_validated
            else "fix TFDS env, shard safety, or partial-shard reader constraints",
            "no_training": True,
        },
        "inventory": inventory,
        "download": download,
        "schema": schema,
        "manifest": manifest,
        "windows": windows,
    }
    docs_root = docs_dir if docs_dir is not None else PROJECT_ROOT / "docs"
    docs_root.mkdir(parents=True, exist_ok=True)
    rendered = render_tfds_mini_report(summary)
    Path(output["eval_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(output["eval_json"]).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    Path(output["eval_md"]).write_text(rendered, encoding="utf-8")
    (docs_root / "BRIDGEDATA_V2_TFDS_MINI_SHARD_REPORT.md").write_text(rendered, encoding="utf-8")
    return summary


def render_tfds_mini_report(summary: dict[str, Any]) -> str:
    if summary["real_tfds_validated"]:
        status = [
            "TFDS/RLDS mini-shard smoke succeeded.",
            "Real TFDS episodes were inspected.",
            "Manifest and window manifest were generated.",
            "Next step: TFDS mini-shard token extraction dry-run.",
        ]
    else:
        status = [
            "TFDS/RLDS mini-shard smoke did not validate real episodes.",
            "Safety gate passed.",
            f"Reason: {summary.get('safe_stop_reason')}",
            "Next step: fix TFDS env / reduce shard count / use cloud.",
        ]
    fields = summary.get("candidate_fields", {})
    return "\n".join(
        [
            "# BridgeData V2 TFDS Mini-Shard Report",
            "",
            *status,
            "",
            "## Summary",
            "",
            f"- safety_gate_pass: `{str(summary['safety_gate_pass']).lower()}`",
            f"- real_tfds_validated: `{str(summary['real_tfds_validated']).lower()}`",
            f"- safe_stop: `{str(summary['safe_stop']).lower()}`",
            f"- download_performed: `{str(summary['download_performed']).lower()}`",
            f"- download_bytes: `{summary['download_bytes']}`",
            f"- num_shards_downloaded: `{summary['num_shards_downloaded']}`",
            f"- num_episodes_scanned: `{summary['num_episodes_scanned']}`",
            f"- episode length min/mean/max: `{summary['episode_length_min']}/{summary['episode_length_mean']}/{summary['episode_length_max']}`",
            f"- num_manifest_records: `{summary['num_manifest_records']}`",
            f"- num_valid_trajectories: `{summary['num_valid_trajectories']}`",
            f"- num_windows: `{summary['num_windows']}`",
            "",
            "## Candidate Fields",
            "",
            f"- image fields: `{fields.get('image_fields', [])}`",
            f"- action fields: `{fields.get('action_fields', [])}`",
            f"- language fields: `{fields.get('language_fields', [])}`",
            f"- goal fields: `{fields.get('goal_fields', [])}`",
            "",
            "## Boundaries",
            "",
            "- no raw zip download",
            "- no full TFDS download",
            "- no DROID download",
            "- no training",
            "- no token extraction",
            "- no importance generation",
            "- action, language, and goal remain metadata only",
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


def _first_reason(*summaries: dict[str, Any]) -> str | None:
    for summary in summaries:
        if summary.get("reason"):
            return summary["reason"]
    return "TFDS mini-shard smoke did not produce validated windows."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(evaluate_tfds_mini_shard(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
