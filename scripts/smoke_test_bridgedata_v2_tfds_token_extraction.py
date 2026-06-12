"""Step24 smoke gate for BridgeData TFDS local-only token extraction."""

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

from eval.eval_bridgedata_v2_tfds_token_extraction import evaluate_step24_token_extraction

CONFIG_PATH = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_token_extraction_step24.yaml"
TARGETED_TESTS = [
    "tests/test_bridgedata_v2_tfds_token_extraction_config.py",
    "tests/test_bridgedata_v2_tfds_clip_cache_fake.py",
    "tests/test_bridgedata_v2_tfds_frame_exporter_fake.py",
    "tests/test_bridgedata_v2_tfds_token_manifest_fake.py",
    "tests/test_bridgedata_v2_tfds_token_extraction_no_download.py",
    "tests/test_bridgedata_v2_tfds_token_extraction_report.py",
    "tests/test_bridgedata_v2_tfds_token_extraction_artifact_blacklist.py",
]


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(CONFIG_PATH)
    env_guard = _env_guard()
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        raise RuntimeError(f"env_isaaclab must not contain TensorFlow/TFDS: {env_guard}")

    _run([sys.executable, "-m", "pytest", *TARGETED_TESTS, "-q"])
    _validate_resolved_fields(config)
    _run(
        [
            str(config["input"]["tfds_env_python"]),
            "scripts/export_bridgedata_v2_tfds_resolved_clips.py",
            "--config",
            str(CONFIG_PATH),
        ]
    )
    _run([sys.executable, "scripts/extract_bridgedata_v2_tfds_videomae_tokens.py", "--config", str(CONFIG_PATH)])
    eval_summary = evaluate_step24_token_extraction(CONFIG_PATH)
    _check_artifact_safety(config, eval_summary)

    print(f"BRIDGEDATA_V2_TFDS_TOKEN_EXTRACTION_PASS = {str(eval_summary['pass']).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_TOKEN_EXTRACTION_SAFE_STOP = {str(eval_summary['safe_stop']).lower()}")
    print(
        "BRIDGEDATA_V2_TFDS_TOKEN_EXTRACTION_SAFETY_GATE_PASS = "
        f"{str(eval_summary['safety_gate_pass']).lower()}"
    )
    if not eval_summary["safety_gate_pass"]:
        raise RuntimeError("Step24 safety gate failed")


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


def _validate_resolved_fields(config: dict[str, Any]) -> None:
    resolved_path = Path(config["input"]["resolved_fields_json"])
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    if resolved.get("image_field") != "steps/observation/image_0":
        raise ValueError(f"Unexpected image_field: {resolved.get('image_field')!r}")
    if bool(resolved.get("image_field_is_metadata_flag")):
        raise ValueError("Step24 refuses metadata image flags")


def _check_artifact_safety(config: dict[str, Any], eval_summary: dict[str, Any]) -> None:
    run_name = config["output"]["run_name"]
    forbidden_suffixes = (".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".mkv")
    run_dir = Path(config["output"]["run_root"]) / run_name
    if run_dir.exists():
        bad = [str(path) for path in run_dir.rglob("*") if path.suffix.lower() in forbidden_suffixes]
        if bad:
            raise RuntimeError(f"Step24 emitted image/video files: {bad[:5]}")
    if eval_summary.get("model_download_performed"):
        raise RuntimeError("Step24 model download flag is true")
    if eval_summary.get("training_performed") or eval_summary.get("importance_generation_performed"):
        raise RuntimeError("Step24 training or importance flag is true")


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
