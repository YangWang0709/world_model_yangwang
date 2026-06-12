"""Prepare Step34 proxy-supervised temporal selector scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_tfds_proxy_temporal_selector_step34 import prepare_step34_proxy_temporal_selector

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_proxy_temporal_selector_step34.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-dry-run", action="store_true")
    args = parser.parse_args()
    summary = prepare_step34_proxy_temporal_selector(args.config, run_dry_run=not args.skip_dry_run)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
