"""Create structured toy token shards for Step 5.5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.structured_token_toy import StructuredTokenToyConfig, write_structured_token_shards


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/structured_token_toy.yaml")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def config_from_mapping(config: dict[str, Any]) -> tuple[StructuredTokenToyConfig, Path, bool]:
    data_cfg = dict(config.get("data", {}))
    output_cfg = dict(config.get("output", {}))
    toy_cfg = StructuredTokenToyConfig(
        num_samples=int(data_cfg.get("num_samples", 32)),
        num_tokens=int(data_cfg.get("num_tokens", 196)),
        token_dim=int(data_cfg.get("token_dim", 768)),
        num_key_tokens=int(data_cfg.get("num_key_tokens", 4)),
        signal_scale=float(data_cfg.get("signal_scale", 3.0)),
        noise_scale=float(data_cfg.get("noise_scale", 0.05)),
        seed=int(config.get("seed", data_cfg.get("seed", 42))),
        split=str(data_cfg.get("split", "structured_toy")),
        shard_size=int(data_cfg.get("shard_size", 8)),
    )
    output_dir = Path(output_cfg.get("token_shard_dir", data_cfg.get("token_shard_dir")))
    overwrite = bool(output_cfg.get("overwrite", True))
    return toy_cfg, output_dir, overwrite


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    toy_config, output_dir, overwrite = config_from_mapping(config)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    summary = write_structured_token_shards(output_dir, toy_config, overwrite=overwrite)
    summary_path = Path(config.get("output", {}).get("summary_path", output_dir / "structured_token_summary.json"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("STRUCTURED_TOKEN_TOY_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
