"""Step29 smoke gate for BridgeData TFDS train-val context utility sanity."""

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

from eval.eval_bridgedata_v2_tfds_trainval_context_utility import evaluate_step29_trainval_context_utility
from scripts.extract_bridgedata_v2_tfds_32_64_tokens import extract_step29_tokens
from scripts.generate_bridgedata_v2_tfds_32_64_importance import generate_step29_importance
from scripts.prepare_bridgedata_v2_tfds_32_64_windows import prepare_step29_windows
from scripts.train_eval_bridgedata_v2_tfds_context_utility import train_eval_step29_context_utility

CONFIG_PATH = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_trainval_context_utility_step29.yaml"
TARGETED_TESTS = [
    "tests/test_bridgedata_v2_tfds_window_sampler_fake.py",
    "tests/test_bridgedata_v2_tfds_trainval_split_fake.py",
    "tests/test_bridgedata_v2_tfds_trainval_context_utility_config.py",
    "tests/test_bridgedata_v2_tfds_expanded_token_manifest_fake.py",
    "tests/test_bridgedata_v2_tfds_expanded_importance_manifest_fake.py",
    "tests/test_bridgedata_v2_tfds_context_utility_metrics_fake.py",
    "tests/test_bridgedata_v2_tfds_trainval_context_utility_no_forbidden_work.py",
    "tests/test_bridgedata_v2_tfds_trainval_context_utility_report.py",
    "tests/test_bridgedata_v2_tfds_trainval_context_utility_artifact_blacklist.py",
]


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(CONFIG_PATH)
    _assert_env_guard()
    _assert_existing_inputs(config)
    _run([sys.executable, "-m", "pytest", *TARGETED_TESTS, "-q"])
    prepare = prepare_step29_windows(CONFIG_PATH)
    if prepare.get("safe_stop"):
        eval_summary = evaluate_step29_trainval_context_utility(CONFIG_PATH)
        _print_result(eval_summary)
        return
    extract = extract_step29_tokens(CONFIG_PATH)
    if extract.get("safe_stop"):
        eval_summary = evaluate_step29_trainval_context_utility(CONFIG_PATH)
        _print_result(eval_summary)
        return
    importance = generate_step29_importance(CONFIG_PATH)
    if importance.get("safe_stop"):
        eval_summary = evaluate_step29_trainval_context_utility(CONFIG_PATH)
        _print_result(eval_summary)
        return
    train_eval_step29_context_utility(CONFIG_PATH)
    eval_summary = evaluate_step29_trainval_context_utility(CONFIG_PATH)
    _check_artifact_safety(config, eval_summary)
    _print_result(eval_summary)
    if not eval_summary["safety_gate_pass"]:
        raise RuntimeError("Step29 safety gate failed")
    if not eval_summary["pass"] and not eval_summary["safe_stop"]:
        raise RuntimeError("Step29 failed without a safe-stop")


def _print_result(eval_summary: dict[str, Any]) -> None:
    print(f"BRIDGEDATA_V2_TFDS_TRAINVAL_CONTEXT_UTILITY_PASS = {str(eval_summary['pass']).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_TRAINVAL_CONTEXT_UTILITY_SAFE_STOP = {str(eval_summary['safe_stop']).lower()}")
    print(
        "BRIDGEDATA_V2_TFDS_TRAINVAL_CONTEXT_UTILITY_SAFETY_GATE_PASS = "
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


def _assert_existing_inputs(config: dict[str, Any]) -> None:
    required = [
        config["input"]["tfds_env_python"],
        config["input"]["tfds_dataset_root"],
        config["input"]["resolved_fields_json"],
        config["input"]["resolved_window_manifest_jsonl"],
        config["input"]["local_videomae_model"],
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"Step29 requires existing inputs and must not download new ones: {missing}")


def _check_artifact_safety(config: dict[str, Any], eval_summary: dict[str, Any]) -> None:
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    forbidden_suffixes = (".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".mkv", ".ckpt", ".safetensors", ".log")
    if run_dir.exists():
        bad = [str(path) for path in run_dir.rglob("*") if path.suffix.lower() in forbidden_suffixes]
        if bad:
            raise RuntimeError(f"Step29 emitted forbidden run artifacts: {bad[:5]}")
    output_text = "\n".join(str(value) for value in config["output"].values())
    if "data/importance_shards" in output_text or "data/token_shards" in output_text:
        raise RuntimeError("Step29 output paths must stay out of data shard directories")
    for key in (
        "new_tfds_shard_downloaded",
        "model_download_performed",
        "videomae_training_performed",
        "teacher_training_performed",
        "selector_training_performed",
        "current_importance_training_performed",
        "data_token_shards_written",
        "data_importance_shards_written",
        "checkpoint_saved",
    ):
        if eval_summary.get(key):
            raise RuntimeError(f"Step29 safety flag unexpectedly true: {key}")
    if eval_summary.get("optimizer_step_scope") != "tiny_world_model_predictor_only":
        raise RuntimeError("Step29 optimizer scope is not tiny predictor only")
    if eval_summary.get("context_utility_claim_allowed"):
        raise RuntimeError("Step29 must not allow a final context utility claim")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


if __name__ == "__main__":
    main()

