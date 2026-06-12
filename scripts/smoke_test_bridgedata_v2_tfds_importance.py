"""Step25 smoke gate for BridgeData TFDS predictive-importance dry-run."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bridgedata_v2_tfds_importance import evaluate_step25_importance

CONFIG_PATH = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_importance_step25.yaml"
TARGETED_TESTS = [
    "tests/test_bridgedata_v2_tfds_importance_config.py",
    "tests/test_bridgedata_v2_tfds_token_artifact_loader_fake.py",
    "tests/test_bridgedata_v2_tfds_importance_proxy_fake.py",
    "tests/test_bridgedata_v2_tfds_importance_manifest_fake.py",
    "tests/test_bridgedata_v2_tfds_importance_no_training_no_download.py",
    "tests/test_bridgedata_v2_tfds_importance_report.py",
    "tests/test_bridgedata_v2_tfds_importance_artifact_blacklist.py",
]


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(CONFIG_PATH)
    env_guard = _env_guard()
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        raise RuntimeError(f"env_isaaclab must not contain TensorFlow/TFDS: {env_guard}")

    _run([sys.executable, "-m", "pytest", *TARGETED_TESTS, "-q"])
    _step24_inputs_exist(config)
    _run([sys.executable, "scripts/generate_bridgedata_v2_tfds_importance_dryrun.py", "--config", str(CONFIG_PATH)])
    eval_summary = evaluate_step25_importance(CONFIG_PATH)
    _check_artifact_safety(config, eval_summary)

    print(f"BRIDGEDATA_V2_TFDS_IMPORTANCE_PASS = {str(eval_summary['pass']).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_IMPORTANCE_SAFE_STOP = {str(eval_summary['safe_stop']).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_IMPORTANCE_SAFETY_GATE_PASS = {str(eval_summary['safety_gate_pass']).lower()}")
    if not eval_summary["safety_gate_pass"]:
        raise RuntimeError("Step25 safety gate failed")
    if not eval_summary["pass"] and not eval_summary["safe_stop"]:
        raise RuntimeError("Step25 failed without a safe-stop")


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


def _step24_inputs_exist(config: dict[str, Any]) -> bool:
    required = [
        Path(config["input"]["token_manifest_jsonl"]),
        Path(config["input"]["token_summary_json"]),
        Path(config["input"]["token_smoke_dir"]),
    ]
    return all(path.exists() for path in required)


def _check_artifact_safety(config: dict[str, Any], eval_summary: dict[str, Any]) -> None:
    forbidden_suffixes = (".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".mkv", ".npz", ".npy")
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    if run_dir.exists():
        bad = [str(path) for path in run_dir.rglob("*") if path.suffix.lower() in forbidden_suffixes]
        if bad:
            raise RuntimeError(f"Step25 emitted forbidden non-importance artifacts: {bad[:5]}")
    output_text = "\n".join(str(value) for value in config["output"].values())
    if "data/importance_shards" in output_text or "data/token_shards" in output_text:
        raise RuntimeError("Step25 output paths must stay out of data shard directories")
    for key in (
        "download_performed",
        "model_download_performed",
        "training_performed",
        "teacher_training_performed",
        "selector_training_performed",
        "world_model_training_performed",
        "token_extraction_performed",
        "large_importance_shards_generated",
        "data_importance_shards_written",
        "data_token_shards_written",
    ):
        if eval_summary.get(key):
            raise RuntimeError(f"Step25 safety flag unexpectedly true: {key}")


def _env_guard() -> dict[str, bool]:
    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


if __name__ == "__main__":
    main()
