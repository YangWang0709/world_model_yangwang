"""Smoke gate for Step19 long-context dataset feasibility."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_long_context_dataset_feasibility import write_feasibility_summary
from scripts.dryrun_bridgedata_v2_schema import run_bridgedata_dryrun
from scripts.dryrun_droid_schema import run_droid_dryrun
from scripts.inspect_long_context_dataset_candidates import inspect_candidates

RUN_DIR = PROJECT_ROOT / "runs" / "long_context_dataset_feasibility_step19_v1"


def _run_pytest(paths: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *paths, "-q"],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
    return {
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "output_tail": "\n".join(output.strip().splitlines()[-40:]),
    }


def main() -> None:
    schema_tests = _run_pytest(
        [
            "tests/test_long_context_dataset_feasibility_config.py",
            "tests/test_long_context_window_spec.py",
            "tests/test_long_context_dataset_schema.py",
        ]
    )
    matrix = inspect_candidates(output_dir=RUN_DIR)
    bridge = run_bridgedata_dryrun(output_dir=RUN_DIR)
    droid = run_droid_dryrun(output_dir=RUN_DIR)
    summary = write_feasibility_summary(run_dir=RUN_DIR)
    passed = bool(
        schema_tests["passed"]
        and matrix["no_download"]
        and bridge["no_download"]
        and droid["no_download"]
        and summary["sanity_gate_pass"]
        and not summary["training_performed"]
        and not summary["full_download_performed"]
        and not summary["large_download_performed"]
    )
    print("# Step19 Smoke Summary")
    print(json.dumps({"schema_tests": schema_tests, "sanity_gate_pass": summary["sanity_gate_pass"]}, indent=2))
    print(f"LONG_CONTEXT_DATASET_FEASIBILITY_PASS = {str(passed).lower()}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
