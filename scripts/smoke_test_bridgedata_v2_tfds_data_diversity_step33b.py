"""Smoke gate for Step33B BridgeData TFDS data-diversity diagnostics."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bridgedata_v2_tfds_data_diversity_step33b import evaluate_step33b_data_diversity
from scripts.extract_bridgedata_v2_tfds_data_diversity_tokens_step33b import extract_step33b_data_diversity_tokens
from scripts.generate_bridgedata_v2_tfds_data_diversity_importance_step33b import (
    generate_step33b_data_diversity_importance,
)
from scripts.prepare_bridgedata_v2_tfds_data_diversity_step33b import prepare_step33b_data_diversity
from scripts.train_eval_bridgedata_v2_tfds_data_diversity_step33b import train_eval_step33b_data_diversity

CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_data_diversity_step33b.yaml"


def main() -> None:
    tests_ok = _run_targeted_tests()
    env_guard = {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }
    artifacts_ok = _required_artifacts_exist()
    safety_gate = tests_ok and artifacts_ok and not env_guard["tensorflow"] and not env_guard["tensorflow_datasets"]
    passed = False
    safe_stop = True
    if safety_gate:
        prep = prepare_step33b_data_diversity(CONFIG)
        safe_stop = bool(prep.get("safe_stop"))
        if not safe_stop:
            token_summary = extract_step33b_data_diversity_tokens(CONFIG)
            if bool(token_summary.get("safe_stop")):
                safe_stop = True
            else:
                importance_summary = generate_step33b_data_diversity_importance(CONFIG)
                if bool(importance_summary.get("safe_stop")):
                    safe_stop = True
                else:
                    train_eval_step33b_data_diversity(CONFIG)
                    result = evaluate_step33b_data_diversity(CONFIG)
                    passed = bool(result.get("pass"))
                    safe_stop = bool(result.get("safe_stop"))
                    safety_gate = safety_gate and bool(result.get("safety_gate_pass"))
        else:
            result = evaluate_step33b_data_diversity(CONFIG)
            passed = bool(result.get("pass"))
            safe_stop = bool(result.get("safe_stop"))
            safety_gate = safety_gate and bool(result.get("safety_gate_pass"))
    print(f"BRIDGEDATA_V2_TFDS_DATA_DIVERSITY_PASS = {str(passed).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_DATA_DIVERSITY_SAFE_STOP = {str(safe_stop).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_DATA_DIVERSITY_SAFETY_GATE_PASS = {str(safety_gate).lower()}")
    if not safety_gate or (not passed and not safe_stop):
        raise SystemExit(1)
    if safe_stop:
        raise SystemExit(1)


def _run_targeted_tests() -> bool:
    tests = [
        "tests/test_bridgedata_v2_tfds_data_diversity_config.py",
        "tests/test_bridgedata_v2_tfds_second_shard_acquisition_fake.py",
        "tests/test_bridgedata_v2_tfds_data_diversity_windows_fake.py",
        "tests/test_bridgedata_v2_tfds_data_diversity_splits_fake.py",
        "tests/test_bridgedata_v2_tfds_data_diversity_proxy_importance_fake.py",
        "tests/test_bridgedata_v2_tfds_cross_shard_metrics_fake.py",
        "tests/test_bridgedata_v2_tfds_dataset_bias_metrics_fake.py",
        "tests/test_bridgedata_v2_tfds_data_diversity_no_forbidden_work.py",
        "tests/test_bridgedata_v2_tfds_data_diversity_report.py",
        "tests/test_bridgedata_v2_tfds_data_diversity_artifact_blacklist.py",
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
        config["input"]["local_videomae_model"],
        config["input"]["shard0_path"],
        config["input"]["step32_horizon_window_manifest_jsonl"],
        config["input"]["step32_horizon_splits_json"],
        config["input"]["step32_frame_repeat_token_manifest_jsonl"],
        config["input"]["step32_importance_manifest_jsonl"],
        config["input"]["step32_decision_json"],
        config["input"]["step33a_decision_json"],
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        print(json.dumps({"missing_step33b_inputs": missing}, indent=2, sort_keys=True))
    return not missing


if __name__ == "__main__":
    main()
