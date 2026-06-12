"""Smoke gate for Step31 teacher temporal/horizon diagnostics."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bridgedata_v2_teacher_temporal_horizon_step31 import evaluate_step31_teacher_temporal_horizon
from scripts.run_bridgedata_v2_teacher_architecture_diagnosis_step31 import run_step31_teacher_architecture_diagnosis
from scripts.run_bridgedata_v2_temporal_horizon_diagnosis_step31 import run_step31_temporal_horizon_diagnosis

CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_teacher_temporal_horizon_diagnosis_step31.yaml"


def main() -> None:
    tests_ok = _run_targeted_tests()
    env_guard = {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }
    artifacts_ok = _required_artifacts_exist()
    safety_gate = tests_ok and artifacts_ok and not env_guard["tensorflow"] and not env_guard["tensorflow_datasets"]
    if safety_gate:
        run_step31_teacher_architecture_diagnosis(CONFIG)
        run_step31_temporal_horizon_diagnosis(CONFIG)
        result = evaluate_step31_teacher_temporal_horizon(CONFIG)
        passed = bool(result.get("pass"))
        safe_stop = bool(result.get("safe_stop"))
        safety_gate = safety_gate and bool(result.get("safety_gate_pass"))
    else:
        passed = False
        safe_stop = True
    print(f"BRIDGEDATA_V2_TFDS_TEACHER_TEMPORAL_HORIZON_PASS = {str(passed).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_TEACHER_TEMPORAL_HORIZON_SAFE_STOP = {str(safe_stop).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_TEACHER_TEMPORAL_HORIZON_SAFETY_GATE_PASS = {str(safety_gate).lower()}")
    if not safety_gate or not passed:
        raise SystemExit(1)


def _run_targeted_tests() -> bool:
    tests = [
        "tests/test_bridgedata_v2_teacher_temporal_horizon_config.py",
        "tests/test_bridgedata_v2_teacher_diagnostic_variants_fake.py",
        "tests/test_bridgedata_v2_teacher_diagnostic_scores_fake.py",
        "tests/test_bridgedata_v2_temporal_horizon_diagnosis_fake.py",
        "tests/test_bridgedata_v2_current_dominance_diagnosis_fake.py",
        "tests/test_bridgedata_v2_teacher_failure_analysis_fake.py",
        "tests/test_bridgedata_v2_teacher_temporal_horizon_no_forbidden_work.py",
        "tests/test_bridgedata_v2_teacher_temporal_horizon_report.py",
        "tests/test_bridgedata_v2_teacher_temporal_horizon_artifact_blacklist.py",
    ]
    result = subprocess.run([sys.executable, "-m", "pytest", *tests, "-q"], cwd=PROJECT_ROOT, text=True)
    return result.returncode == 0


def _required_artifacts_exist() -> bool:
    import yaml

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    required = [
        config["input"]["token_manifest_jsonl"],
        config["input"]["token_summary_json"],
        config["input"]["token_smoke_dir"],
        config["input"]["proxy_importance_manifest_jsonl"],
        config["input"]["proxy_importance_summary_json"],
        config["input"]["proxy_importance_dir"],
        config["input"]["multiseed_splits_json"],
        config["input"]["teacher_train_summary_json"],
        config["input"]["teacher_importance_manifest_jsonl"],
        config["input"]["teacher_importance_summary_json"],
        config["input"]["teacher_proxy_comparison_json"],
        config["input"]["teacher_topk_utility_json"],
        config["input"]["teacher_decision_json"],
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        print(json.dumps({"missing_step31_inputs": missing}, indent=2, sort_keys=True))
    return not missing


if __name__ == "__main__":
    main()
