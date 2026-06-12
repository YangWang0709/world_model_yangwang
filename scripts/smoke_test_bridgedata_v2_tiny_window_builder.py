"""Smoke gate for Step20 BridgeData V2 tiny-window builder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bridgedata_v2_window_builder import evaluate_window_builder
from scripts.build_bridgedata_v2_context_windows import build_from_config
from scripts.inspect_bridgedata_v2_tiny_subset import inspect_from_config

RUN_DIR = PROJECT_ROOT / "runs" / "bridgedata_v2_tiny_window_builder_step20_v1"
TARGETED_TESTS = [
    "tests/test_bridgedata_v2_tiny_window_builder_config.py",
    "tests/test_bridgedata_v2_manifest_schema.py",
    "tests/test_bridgedata_v2_window_builder_fake_manifest.py",
    "tests/test_bridgedata_v2_missing_local_subset_ok.py",
    "tests/test_bridgedata_v2_no_download_guard.py",
    "tests/test_bridgedata_v2_artifact_blacklist.py",
    "tests/test_bridgedata_v2_window_builder_report.py",
]


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
        "output_tail": "\n".join(output.strip().splitlines()[-50:]),
    }


def main() -> None:
    targeted = _run_pytest(TARGETED_TESTS)
    inspection = inspect_from_config()
    builder = build_from_config()
    evaluation = evaluate_window_builder()
    pt_outputs = sorted(str(path) for path in RUN_DIR.glob("**/*.pt"))
    pass_flag = bool(
        targeted["passed"]
        and inspection["no_download"]
        and builder["sanity_gate_pass"]
        and evaluation["pass"]
        and Path(builder["window_manifest_jsonl"]).exists()
        and (PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TINY_WINDOW_BUILDER_REPORT.md").exists()
        and not pt_outputs
    )
    print("# Step20 Smoke Summary")
    print(
        json.dumps(
            {
                "targeted_tests": targeted,
                "local_subset_exists": inspection["local_subset_exists"],
                "used_fake_manifest": builder["used_fake_manifest"],
                "num_windows": builder["num_windows"],
                "pt_outputs": pt_outputs,
                "eval_pass": evaluation["pass"],
            },
            indent=2,
        )
    )
    print(f"BRIDGEDATA_V2_TINY_WINDOW_BUILDER_PASS = {str(pass_flag).lower()}")
    if not pass_flag:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
