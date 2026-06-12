"""Step26 smoke gate for BridgeData TFDS world-model forward/loss."""

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

from eval.eval_bridgedata_v2_tfds_world_model_smoke import evaluate_step26_world_model_smoke

CONFIG_PATH = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_world_model_smoke_step26.yaml"
TARGETED_TESTS = [
    "tests/test_bridgedata_v2_tfds_world_model_smoke_config.py",
    "tests/test_bridgedata_v2_tfds_world_model_batch_fake.py",
    "tests/test_bridgedata_v2_tfds_context_selection_fake.py",
    "tests/test_bridgedata_v2_tfds_world_model_forward_fake.py",
    "tests/test_bridgedata_v2_tfds_world_model_metrics_fake.py",
    "tests/test_bridgedata_v2_tfds_world_model_no_training_no_download.py",
    "tests/test_bridgedata_v2_tfds_world_model_report.py",
    "tests/test_bridgedata_v2_tfds_world_model_artifact_blacklist.py",
]


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(CONFIG_PATH)
    env_guard = _env_guard()
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        raise RuntimeError(f"env_isaaclab must not contain TensorFlow/TFDS: {env_guard}")

    _run([sys.executable, "-m", "pytest", *TARGETED_TESTS, "-q"])
    _run([sys.executable, "scripts/run_bridgedata_v2_tfds_world_model_smoke.py", "--config", str(CONFIG_PATH)])
    eval_summary = evaluate_step26_world_model_smoke(CONFIG_PATH)
    _check_artifact_safety(config, eval_summary)

    print(f"BRIDGEDATA_V2_TFDS_WORLD_MODEL_SMOKE_PASS = {str(eval_summary['pass']).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_WORLD_MODEL_SMOKE_SAFE_STOP = {str(eval_summary['safe_stop']).lower()}")
    print(
        "BRIDGEDATA_V2_TFDS_WORLD_MODEL_SMOKE_SAFETY_GATE_PASS = "
        f"{str(eval_summary['safety_gate_pass']).lower()}"
    )
    if not eval_summary["safety_gate_pass"]:
        raise RuntimeError("Step26 safety gate failed")
    if not eval_summary["pass"] and not eval_summary["safe_stop"]:
        raise RuntimeError("Step26 failed without a safe-stop")


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
            raise RuntimeError(f"Step26 emitted forbidden artifacts: {bad[:5]}")
    output_text = "\n".join(str(value) for value in config["output"].values())
    if "data/importance_shards" in output_text or "data/token_shards" in output_text:
        raise RuntimeError("Step26 output paths must stay out of data shard directories")
    for key in (
        "download_performed",
        "model_download_performed",
        "training_performed",
        "optimizer_step_performed",
        "teacher_training_performed",
        "selector_training_performed",
        "world_model_training_performed",
        "token_extraction_performed",
        "importance_generation_performed",
        "data_token_shards_written",
        "data_importance_shards_written",
    ):
        if eval_summary.get(key):
            raise RuntimeError(f"Step26 safety flag unexpectedly true: {key}")


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
