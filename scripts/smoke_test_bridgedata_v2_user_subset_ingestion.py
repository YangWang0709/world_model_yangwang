"""Smoke gate for Step22 BridgeData V2 user-subset ingestion."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bridgedata_v2_user_subset_ingestion import evaluate_user_subset_ingestion
from scripts.build_bridgedata_v2_user_subset_manifest import build_manifest_from_config
from scripts.build_bridgedata_v2_user_subset_windows import build_user_subset_windows_from_config
from scripts.create_bridgedata_v2_user_subset_template import run_from_config as create_template_from_config
from scripts.inspect_bridgedata_v2_user_subset import inspect_from_config

RUN_DIR = PROJECT_ROOT / "runs" / "bridgedata_v2_user_subset_ingestion_step22_v1"
TARGETED_TESTS = [
    "tests/test_bridgedata_v2_user_subset_ingestion_config.py",
    "tests/test_bridgedata_v2_user_subset_template.py",
    "tests/test_bridgedata_v2_user_subset_manifest_tools.py",
    "tests/test_bridgedata_v2_user_subset_validator_missing_ok.py",
    "tests/test_bridgedata_v2_user_subset_validator_fake_realistic.py",
    "tests/test_bridgedata_v2_user_subset_windows_from_manifest.py",
    "tests/test_bridgedata_v2_user_subset_report.py",
    "tests/test_bridgedata_v2_user_subset_artifact_blacklist.py",
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
        "output_tail": "\n".join(output.strip().splitlines()[-60:]),
    }


def main() -> None:
    targeted = _run_pytest(TARGETED_TESTS)
    template = create_template_from_config()
    inspection = inspect_from_config()
    manifest = build_manifest_from_config()
    window = build_user_subset_windows_from_config()
    evaluation = evaluate_user_subset_ingestion()
    pt_outputs = sorted(str(path) for path in RUN_DIR.glob("**/*.pt"))
    image_outputs = sorted(
        str(path)
        for suffix in ("*.jpg", "*.jpeg", "*.png", "*.mp4", "*.avi", "*.mov", "*.mkv")
        for path in RUN_DIR.glob(f"**/{suffix}")
    )
    safety_pass = bool(
        targeted["passed"]
        and evaluation["safety_gate_pass"]
        and not evaluation["download_performed"]
        and not evaluation["training_performed"]
        and not evaluation["token_extraction_performed"]
        and not evaluation["importance_generation_performed"]
        and not pt_outputs
        and not image_outputs
        and template["text_only"]
    )
    print("# Step22 Smoke Summary")
    print(
        json.dumps(
            {
                "targeted_tests": targeted,
                "template": template,
                "inspection_pending": inspection["pending_user_data"],
                "manifest_records": manifest["num_manifest_records"],
                "real_format_validated": window["real_format_validated"],
                "num_windows": window["num_windows"],
                "pt_outputs": pt_outputs,
                "image_outputs": image_outputs,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"BRIDGEDATA_V2_USER_SUBSET_PENDING = {str(evaluation['pending_user_data']).lower()}")
    print(f"BRIDGEDATA_V2_USER_SUBSET_VALIDATED = {str(evaluation['real_format_validated']).lower()}")
    print(f"BRIDGEDATA_V2_USER_SUBSET_SAFETY_GATE_PASS = {str(safety_pass).lower()}")
    if not safety_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
