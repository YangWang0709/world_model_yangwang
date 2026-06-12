"""Smoke gate for Step32 BridgeData longer-horizon diagnostics."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bridgedata_v2_tfds_longer_horizon_step32 import evaluate_step32_longer_horizon
from scripts.extract_bridgedata_v2_tfds_longer_horizon_tokens_step32 import extract_step32_longer_horizon_tokens
from scripts.generate_bridgedata_v2_tfds_longer_horizon_importance_step32 import generate_step32_longer_horizon_importance
from scripts.prepare_bridgedata_v2_tfds_longer_horizon_windows_step32 import prepare_step32_longer_horizon_windows
from scripts.train_eval_bridgedata_v2_tfds_longer_horizon_step32 import train_eval_step32_longer_horizon

CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_longer_horizon_step32.yaml"


def main() -> None:
    tests_ok = _run_targeted_tests()
    env_guard = {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }
    artifacts_ok = _required_artifacts_exist()
    safety_gate = tests_ok and artifacts_ok and not env_guard["tensorflow"] and not env_guard["tensorflow_datasets"]
    if safety_gate:
        prepare = prepare_step32_longer_horizon_windows(CONFIG)
        if bool(prepare.get("safe_stop")):
            passed = False
            safe_stop = True
        else:
            extract_step32_longer_horizon_tokens(CONFIG)
            generate_step32_longer_horizon_importance(CONFIG)
            train_eval_step32_longer_horizon(CONFIG)
            result = evaluate_step32_longer_horizon(CONFIG)
            passed = bool(result.get("pass"))
            safe_stop = bool(result.get("safe_stop"))
            safety_gate = safety_gate and bool(result.get("safety_gate_pass"))
    else:
        passed = False
        safe_stop = True
    print(f"BRIDGEDATA_V2_TFDS_LONGER_HORIZON_PASS = {str(passed).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_LONGER_HORIZON_SAFE_STOP = {str(safe_stop).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_LONGER_HORIZON_SAFETY_GATE_PASS = {str(safety_gate).lower()}")
    if not safety_gate or not passed:
        raise SystemExit(1)


def _run_targeted_tests() -> bool:
    tests = [
        "tests/test_bridgedata_v2_tfds_longer_horizon_config.py",
        "tests/test_bridgedata_v2_tfds_longer_horizon_window_builder_fake.py",
        "tests/test_bridgedata_v2_tfds_longer_horizon_split_fake.py",
        "tests/test_bridgedata_v2_tfds_longer_horizon_manifest_fake.py",
        "tests/test_bridgedata_v2_tfds_longer_horizon_proxy_importance_fake.py",
        "tests/test_bridgedata_v2_tfds_longer_horizon_metrics_fake.py",
        "tests/test_bridgedata_v2_tfds_longer_horizon_no_forbidden_work.py",
        "tests/test_bridgedata_v2_tfds_longer_horizon_report.py",
        "tests/test_bridgedata_v2_tfds_longer_horizon_artifact_blacklist.py",
    ]
    result = subprocess.run([sys.executable, "-m", "pytest", *tests, "-q"], cwd=PROJECT_ROOT, text=True)
    return result.returncode == 0


def _required_artifacts_exist() -> bool:
    import yaml

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    required = [
        config["input"]["tfds_dataset_root"],
        config["input"]["tfds_env_python"],
        config["input"]["resolved_fields_json"],
        config["input"]["resolved_window_manifest_jsonl"],
        config["input"]["local_videomae_model"],
        config["input"]["step31_decision_json"],
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        print(json.dumps({"missing_step32_inputs": missing}, indent=2, sort_keys=True))
    return not missing


if __name__ == "__main__":
    main()
