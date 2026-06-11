"""CLI entrypoint for Step 13 BAIR selector loss ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.run_baseline_comparison import load_yaml
from training.selector_ablation_trainer import run_selector_ablation


def _has_shards(path: str | Path, pattern: str) -> bool:
    return bool(list(Path(path).glob(pattern)))


def validate_selector_ablation_inputs(config: dict[str, Any]) -> None:
    data_cfg = config["data"]
    token_glob = data_cfg.get("token_shard_glob", "tokens_shard_*.pt")
    importance_glob = data_cfg.get("importance_shard_glob", "importance_shard_*.pt")
    required_shards = [
        ("Step 11B train VideoMAE token shards", data_cfg["train_token_shard_dir"], token_glob),
        ("Step 11B test VideoMAE token shards", data_cfg["test_token_shard_dir"], token_glob),
        ("Step 11D train importance shards", data_cfg["train_importance_shard_dir"], importance_glob),
        ("Step 11D test importance shards", data_cfg["test_importance_shard_dir"], importance_glob),
    ]
    for label, shard_dir, pattern in required_shards:
        if not _has_shards(shard_dir, pattern):
            raise FileNotFoundError(f"Missing {label}: {shard_dir} with pattern {pattern}")
    teacher_checkpoint = Path(config["teacher_reference"]["checkpoint"])
    if not teacher_checkpoint.exists():
        raise FileNotFoundError(f"Missing Step 11C Teacher checkpoint: {teacher_checkpoint}")
    step12_summary = Path(config["step12_reference"]["baseline_summary_json"])
    if not step12_summary.exists():
        raise FileNotFoundError(f"Missing Step 12 baseline summary: {step12_summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/selector_ablation_bair_videomae_smoke.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    validate_selector_ablation_inputs(config)
    summary = run_selector_ablation(config)
    print("BAIR_SELECTOR_ABLATION_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
