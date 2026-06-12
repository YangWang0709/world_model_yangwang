"""Step30B smoke gate for trained-predictor occlusion teacher."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_tfds_occlusion_teacher_labels import generate_step30b_occlusion_teacher_labels
from eval.eval_bridgedata_v2_tfds_occlusion_teacher_step30b import evaluate_step30b_occlusion_teacher
from scripts.evaluate_bridgedata_v2_tfds_teacher_topk_step30b import evaluate_step30b_teacher_topk
from training.bridgedata_v2_tfds_occlusion_teacher_trainer import train_step30b_teachers_from_config

CONFIG_PATH = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_occlusion_teacher_step30b.yaml"
TARGETED_TESTS = [
    "tests/test_bridgedata_v2_tfds_occlusion_teacher_config.py",
    "tests/test_bridgedata_v2_trained_predictor_teacher_fake.py",
    "tests/test_bridgedata_v2_tfds_occlusion_label_generation_fake.py",
    "tests/test_bridgedata_v2_tfds_teacher_proxy_comparison_fake.py",
    "tests/test_bridgedata_v2_tfds_teacher_topk_metrics_fake.py",
    "tests/test_bridgedata_v2_tfds_occlusion_teacher_no_forbidden_work.py",
    "tests/test_bridgedata_v2_tfds_occlusion_teacher_report.py",
    "tests/test_bridgedata_v2_tfds_occlusion_teacher_artifact_blacklist.py",
]


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(CONFIG_PATH)
    _assert_env_guard()
    _assert_existing_step30a_inputs(config)
    _run([sys.executable, "-m", "pytest", *TARGETED_TESTS, "-q"])
    bundle = train_step30b_teachers_from_config(CONFIG_PATH)
    if bundle["teacher_train_summary"].get("safe_stop"):
        eval_summary = evaluate_step30b_occlusion_teacher(CONFIG_PATH)
        _print_result(eval_summary)
        return
    labels = generate_step30b_occlusion_teacher_labels(CONFIG_PATH, training_bundle=bundle)
    if labels["teacher_importance_summary"].get("safe_stop"):
        eval_summary = evaluate_step30b_occlusion_teacher(CONFIG_PATH)
        _print_result(eval_summary)
        return
    evaluate_step30b_teacher_topk(CONFIG_PATH)
    eval_summary = evaluate_step30b_occlusion_teacher(CONFIG_PATH)
    _check_safety(eval_summary)
    _print_result(eval_summary)
    if not eval_summary["safety_gate_pass"]:
        raise RuntimeError("Step30B safety gate failed")
    if not eval_summary["pass"] and not eval_summary["safe_stop"]:
        raise RuntimeError("Step30B failed without a safe-stop")


def _print_result(eval_summary: dict[str, Any]) -> None:
    print(f"BRIDGEDATA_V2_TFDS_OCCLUSION_TEACHER_PASS = {str(eval_summary['pass']).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_OCCLUSION_TEACHER_SAFE_STOP = {str(eval_summary['safe_stop']).lower()}")
    print(
        "BRIDGEDATA_V2_TFDS_OCCLUSION_TEACHER_SAFETY_GATE_PASS = "
        f"{str(eval_summary['safety_gate_pass']).lower()}"
    )


def _run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"},
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def _assert_env_guard() -> None:
    guard = {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }
    if guard["tensorflow"] or guard["tensorflow_datasets"]:
        raise RuntimeError(f"env_isaaclab must not contain TensorFlow/TFDS: {guard}")


def _assert_existing_step30a_inputs(config: dict[str, Any]) -> None:
    required = [
        config["input"]["token_manifest_jsonl"],
        config["input"]["token_summary_json"],
        config["input"]["token_smoke_dir"],
        config["input"]["proxy_importance_manifest_jsonl"],
        config["input"]["proxy_importance_summary_json"],
        config["input"]["proxy_importance_dir"],
        config["input"]["multiseed_splits_json"],
        config["input"]["stability_summary_json"],
        config["input"]["context_signal_decision_json"],
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"Step30B requires Step30A artifacts and must not regenerate them: {missing}")


def _check_safety(eval_summary: dict[str, Any]) -> None:
    for key in (
        "new_tfds_shard_downloaded",
        "model_download_performed",
        "token_extraction_performed",
        "proxy_importance_regeneration_performed",
        "videomae_training_performed",
        "context_teacher_large_training_performed",
        "selector_training_performed",
        "current_importance_training_performed",
        "data_token_shards_written",
        "data_importance_shards_written",
        "checkpoint_saved",
        "context_utility_claim_allowed",
        "selector_training_allowed",
        "current_importance_training_allowed",
    ):
        if eval_summary.get(key):
            raise RuntimeError(f"Step30B safety flag unexpectedly true: {key}")
    if eval_summary.get("teacher_training_scope") != "small_predictor_teacher_only":
        raise RuntimeError("Step30B teacher scope must be small predictor only")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


if __name__ == "__main__":
    main()
