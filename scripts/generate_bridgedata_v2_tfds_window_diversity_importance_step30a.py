"""Generate Step30A limited proxy importance from Step30A token artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_bridgedata_v2_tfds_importance_dryrun import (
    generate_bridgedata_v2_tfds_importance_from_config,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_window_diversity_step30a.yaml"


def generate_step30a_window_diversity_importance(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(config_path)
    missing = _missing_inputs(config)
    if missing:
        summary = _safe_stop_summary(config, f"missing Step30A importance inputs: {missing}")
        _write_json(Path(config["output"]["importance_summary_json"]), summary)
        return summary
    adapter = _adapter_config(config)
    adapter_path = Path(config["output"]["adapter_config_yaml"])
    adapter_path.write_text(yaml.safe_dump(adapter, sort_keys=False), encoding="utf-8")
    summary = generate_bridgedata_v2_tfds_importance_from_config(adapter_path)
    summary.update(
        {
            "stage": "bridgedata_v2_tfds_window_diversity_step30a_importance",
            "limited_proxy_importance_generation_performed": bool(summary.get("importance_generation_performed")),
            "num_samples": int(summary.get("num_samples") or 0),
            "current_tokens_kept_full": True,
            "train_current_importance": False,
            "teacher_training_performed": False,
            "selector_training_performed": False,
            "data_importance_shards_written": False,
            "current_importance_generated": False,
            "label_quality_note": "proxy dry-run only; not final teacher label",
        }
    )
    _write_json(Path(config["output"]["importance_summary_json"]), summary)
    return summary


def _adapter_config(config: dict[str, Any]) -> dict[str, Any]:
    adapter = dict(config)
    adapter["input"] = {
        "step24_run_dir": str(Path(config["output"]["token_smoke_dir"]).parent),
        "token_manifest_jsonl": config["output"]["token_manifest_jsonl"],
        "token_summary_json": config["output"]["token_summary_json"],
        "token_smoke_dir": config["output"]["token_smoke_dir"],
    }
    adapter["dry_run_limits"] = dict(config["dry_run_limits"])
    adapter["dry_run_limits"]["max_samples"] = _count_jsonl(config["output"]["token_manifest_jsonl"])
    adapter["output"] = dict(config["output"])
    return adapter


def _missing_inputs(config: dict[str, Any]) -> list[str]:
    required = [
        config["output"]["token_manifest_jsonl"],
        config["output"]["token_summary_json"],
        config["output"]["token_smoke_dir"],
    ]
    return [str(path) for path in required if not Path(path).exists()]


def _safe_stop_summary(config: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "stage": "bridgedata_v2_tfds_window_diversity_step30a_importance",
        "limited_proxy_importance_generation_performed": False,
        "importance_generation_performed": False,
        "safe_stop": True,
        "reason": reason,
        "num_samples": 0,
        "method": "proxy_token_mse_dryrun",
        "context_importance_shape_example": None,
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "data_importance_shards_written": False,
        "label_quality_note": "proxy dry-run only; not final teacher label",
        "safety_gate_pass": True,
    }


def _count_jsonl(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(generate_step30a_window_diversity_importance(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
