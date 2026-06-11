"""CLI entrypoint for Step18 BAIR context-selector oracle-gap diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.context_selector_oracle_gap_trainer import DEFAULT_CONFIG, run_bair_context_selector_oracle_gap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    summary = run_bair_context_selector_oracle_gap(parse_args().config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
