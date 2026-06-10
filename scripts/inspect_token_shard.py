"""Inspect and validate a token shard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.token_shards import load_token_shard, summarize_token_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard = load_token_shard(args.shard, map_location="cpu")
    summary = summarize_token_shard(shard)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

