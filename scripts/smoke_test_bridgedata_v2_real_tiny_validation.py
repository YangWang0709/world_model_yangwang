"""Smoke gate for Step21 BridgeData V2 real tiny validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bridgedata_v2_real_tiny_validation import evaluate_real_tiny_validation
from scripts.acquire_bridgedata_v2_real_tiny_subset import run_acquisition_from_config
from scripts.probe_bridgedata_v2_download_options import run_probe_from_config
from scripts.validate_bridgedata_v2_real_tiny_subset import run_validation_from_config

RUN_DIR = PROJECT_ROOT / "runs" / "bridgedata_v2_real_tiny_validation_step21_v1"
TARGETED_TESTS = [
    "tests/test_bridgedata_v2_real_tiny_validation_config.py",
    "tests/test_bridgedata_v2_download_probe_no_large_download.py",
    "tests/test_bridgedata_v2_acquisition_safe_stop.py",
    "tests/test_bridgedata_v2_manifest_autobuilder_fake_files.py",
    "tests/test_bridgedata_v2_real_format_validator.py",
    "tests/test_bridgedata_v2_real_tiny_eval_report.py",
    "tests/test_bridgedata_v2_real_tiny_artifact_blacklist.py",
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
    probe = run_probe_from_config()
    acquisition = run_acquisition_from_config()
    validation = run_validation_from_config()
    evaluation = evaluate_real_tiny_validation()
    pt_outputs = sorted(str(path) for path in RUN_DIR.glob("**/*.pt"))
    safety_pass = bool(
        targeted["passed"]
        and evaluation["safety_gate_pass"]
        and not evaluation["full_download_performed"]
        and not evaluation["large_download_performed"]
        and not evaluation["droid_download_performed"]
        and not evaluation["training_performed"]
        and not evaluation["token_extraction_performed"]
        and not evaluation["importance_generation_performed"]
        and not pt_outputs
    )
    print("# Step21 Smoke Summary")
    print(
        json.dumps(
            {
                "targeted_tests": targeted,
                "safe_candidate_found": probe["safe_candidate_found"],
                "download_performed": acquisition["download_performed"],
                "real_format_validated": validation["real_format_validated"],
                "safe_stop": evaluation["safe_stop"],
                "num_real_windows": evaluation["num_real_windows"],
                "pt_outputs": pt_outputs,
            },
            indent=2,
        )
    )
    print(f"BRIDGEDATA_V2_REAL_TINY_FORMAT_VALIDATED = {str(evaluation['real_format_validated']).lower()}")
    print(f"BRIDGEDATA_V2_REAL_TINY_SAFE_STOP = {str(evaluation['safe_stop']).lower()}")
    print(f"BRIDGEDATA_V2_REAL_TINY_SAFETY_GATE_PASS = {str(safety_pass).lower()}")
    if not safety_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
