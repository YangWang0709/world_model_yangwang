"""Inspect a Step 5 predictive importance shard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.importance_shards import load_importance_shard, summarize_importance_shard, validate_importance_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True)
    return parser.parse_args()


def inspect_importance_shard(path: str | Path) -> dict:
    shard = load_importance_shard(path, map_location="cpu")
    validate_importance_shard(shard, strict=True)
    return summarize_importance_shard(shard)


def main() -> None:
    args = parse_args()
    print(json.dumps(inspect_importance_shard(args.shard), indent=2))


if __name__ == "__main__":
    main()
