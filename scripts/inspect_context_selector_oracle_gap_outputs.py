"""Inspect Step18 BAIR context-selector oracle-gap outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bair_context_selector_oracle_gap import DEFAULT_RUN_DIR, render_context_selector_oracle_gap_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    return parser.parse_args()


def main() -> None:
    run_dir = Path(parse_args().run_dir)
    summary_path = run_dir / "context_selector_oracle_gap_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"Expected summary object: {summary_path}")
    markdown = render_context_selector_oracle_gap_markdown(summary)
    (run_dir / "context_selector_oracle_gap_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
