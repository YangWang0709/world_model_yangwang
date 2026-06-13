"""Smoke gate for Step37 bounded factorized selector diagnosis."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_factorized_selector_dataset_step37 import (
    load_step37_config,
    missing_step37_inputs,
)
from eval.eval_bridgedata_v2_factorized_selector_step37 import evaluate_step37_factorized_selector
from training.bridgedata_v2_factorized_selector_trainer_step37 import train_step37_factorized_selector

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_factorized_selector_step37.yaml"


def main() -> None:
    tests_ok = _run_targeted_tests()
    config = load_step37_config(DEFAULT_CONFIG)
    env_guard = {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }
    missing = missing_step37_inputs(config)
    if missing:
        print(json.dumps({"missing_step37_inputs": missing}, indent=2, sort_keys=True))
    safety_gate = tests_ok and not env_guard["tensorflow"] and not env_guard["tensorflow_datasets"] and not missing
    passed = False
    safe_stop = True
    if safety_gate:
        payload = train_step37_factorized_selector(DEFAULT_CONFIG)
        safe_stop = bool(payload["train_summary"].get("safe_stop", False))
        result = evaluate_step37_factorized_selector(DEFAULT_CONFIG)
        passed = bool(result.get("pass", False))
        safety_gate = safety_gate and bool(result.get("safety_gate_pass", False))
    print(f"BRIDGEDATA_V2_TFDS_FACTORIZED_SELECTOR_STEP37_PASS = {str(passed).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_FACTORIZED_SELECTOR_STEP37_SAFE_STOP = {str(safe_stop).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_FACTORIZED_SELECTOR_STEP37_SAFETY_GATE_PASS = {str(safety_gate).lower()}")
    if not passed or safe_stop or not safety_gate:
        raise SystemExit(1)


def _run_targeted_tests() -> bool:
    tests = [
        "tests/test_bridgedata_v2_factorized_selector_step37_config.py",
        "tests/test_bridgedata_v2_factorized_selector_model_step37_fake.py",
        "tests/test_bridgedata_v2_factorized_selector_dataset_step37_fake.py",
        "tests/test_bridgedata_v2_factorized_selector_losses_step37_fake.py",
        "tests/test_bridgedata_v2_factorized_selector_metrics_step37_fake.py",
        "tests/test_bridgedata_v2_factorized_selector_trainer_step37_fake.py",
        "tests/test_bridgedata_v2_factorized_selector_step37_no_forbidden_work.py",
        "tests/test_bridgedata_v2_factorized_selector_step37_report.py",
        "tests/test_bridgedata_v2_factorized_selector_step37_artifact_blacklist.py",
    ]
    result = subprocess.run([sys.executable, "-m", "pytest", *tests, "-q"], cwd=PROJECT_ROOT, text=True)
    return result.returncode == 0


if __name__ == "__main__":
    main()
