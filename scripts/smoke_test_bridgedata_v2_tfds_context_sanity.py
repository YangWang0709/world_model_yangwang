"""Step28 smoke gate for BridgeData TFDS context sanity analysis."""

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

from analysis.bridgedata_v2_tfds_context_sanity import missing_required_inputs, run_context_sanity_analysis
from eval.eval_bridgedata_v2_tfds_context_sanity import evaluate_step28_context_sanity

CONFIG_PATH = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_context_sanity_step28.yaml"
TARGETED_TESTS = [
    "tests/test_bridgedata_v2_tfds_context_sanity_config.py",
    "tests/test_bridgedata_v2_tfds_context_sanity_metrics_fake.py",
    "tests/test_bridgedata_v2_tfds_teacher_planning_fake.py",
    "tests/test_bridgedata_v2_tfds_context_sanity_no_training_no_download.py",
    "tests/test_bridgedata_v2_tfds_context_sanity_report.py",
    "tests/test_bridgedata_v2_tfds_context_sanity_artifact_blacklist.py",
]


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(CONFIG_PATH)
    env_guard = _env_guard()
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        raise RuntimeError(f"env_isaaclab must not contain TensorFlow/TFDS: {env_guard}")

    _run([sys.executable, "-m", "pytest", *TARGETED_TESTS, "-q"])
    missing = missing_required_inputs(config)
    if missing:
        summary = run_context_sanity_analysis(CONFIG_PATH)
        eval_summary = evaluate_step28_context_sanity(CONFIG_PATH)
    else:
        summary = run_context_sanity_analysis(CONFIG_PATH)
        eval_summary = evaluate_step28_context_sanity(CONFIG_PATH)
    _check_artifact_safety(config, eval_summary)

    print(f"BRIDGEDATA_V2_TFDS_CONTEXT_SANITY_PASS = {str(eval_summary['pass']).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_CONTEXT_SANITY_SAFE_STOP = {str(eval_summary['safe_stop']).lower()}")
    print(
        "BRIDGEDATA_V2_TFDS_CONTEXT_SANITY_SAFETY_GATE_PASS = "
        f"{str(eval_summary['safety_gate_pass']).lower()}"
    )
    if not eval_summary["safety_gate_pass"]:
        raise RuntimeError("Step28 safety gate failed")
    if not eval_summary["pass"] and not eval_summary["safe_stop"]:
        raise RuntimeError(f"Step28 failed without a safe-stop: {summary.get('summary', {})}")


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


def _check_artifact_safety(config: dict[str, Any], eval_summary: dict[str, Any]) -> None:
    forbidden_suffixes = (
        ".jpg",
        ".jpeg",
        ".png",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".npz",
        ".npy",
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".log",
    )
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    if run_dir.exists():
        bad = [str(path) for path in run_dir.rglob("*") if path.suffix.lower() in forbidden_suffixes]
        if bad:
            raise RuntimeError(f"Step28 emitted forbidden artifacts: {bad[:5]}")
    output_text = "\n".join(str(value) for value in config["output"].values())
    if "data/importance_shards" in output_text or "data/token_shards" in output_text:
        raise RuntimeError("Step28 output paths must stay out of data shard directories")
    for key in (
        "download_performed",
        "model_download_performed",
        "training_performed",
        "optimizer_step_performed",
        "token_extraction_performed",
        "importance_generation_performed",
        "videomae_training_performed",
        "teacher_training_performed",
        "selector_training_performed",
        "current_importance_training_performed",
        "world_model_training_performed",
        "data_token_shards_written",
        "data_importance_shards_written",
        "checkpoint_saved",
    ):
        if eval_summary.get(key):
            raise RuntimeError(f"Step28 safety flag unexpectedly true: {key}")
    if eval_summary.get("train_current_importance_now"):
        raise RuntimeError("Step28 must not train current importance")


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

