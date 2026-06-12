"""Smoke gate for Step23 BridgeData V2 TFDS/RLDS mini-shard ingestion."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bridgedata_v2_tfds_mini_shard import evaluate_tfds_mini_shard
from scripts.build_bridgedata_v2_tfds_mini_manifest import build_manifest_from_config
from scripts.build_bridgedata_v2_tfds_mini_windows import build_windows_from_config
from scripts.download_bridgedata_v2_tfds_mini_shards import download_from_config
from scripts.inspect_bridgedata_v2_rlds_schema import inspect_from_config
from scripts.inventory_bridgedata_v2_tfds_shards import inventory_from_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_mini_shard_step23.yaml"
TARGETED_TESTS = [
    "tests/test_bridgedata_v2_tfds_mini_config.py",
    "tests/test_bridgedata_v2_tfds_inventory_parser.py",
    "tests/test_bridgedata_v2_tfds_safe_downloader_limits.py",
    "tests/test_bridgedata_v2_rlds_schema_inspector_fake.py",
    "tests/test_bridgedata_v2_rlds_to_manifest_fake.py",
    "tests/test_bridgedata_v2_tfds_mini_pending_or_validated_report.py",
    "tests/test_bridgedata_v2_tfds_mini_artifact_blacklist.py",
]


def _run_pytest(paths: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *paths, "-q"],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
    return {
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "output_tail": "\n".join(output.strip().splitlines()[-80:]),
    }


def _check_tfds_env(config: dict[str, Any]) -> dict[str, Any]:
    tfds_python = Path(config["paths"]["tfds_env_python"])
    if not tfds_python.exists():
        return {"exists": False, "tensorflow": False, "tensorflow_datasets": False}
    result = subprocess.run(
        [
            str(tfds_python),
            "-c",
            "import importlib.util,json; print(json.dumps({"
            "'exists': True, "
            "'tensorflow': importlib.util.find_spec('tensorflow') is not None, "
            "'tensorflow_datasets': importlib.util.find_spec('tensorflow_datasets') is not None"
            "}))",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return {"exists": True, "tensorflow": False, "tensorflow_datasets": False, "error": result.stderr.strip()}
    return json.loads(result.stdout)


def main() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    run_dir = Path(config["output"]["eval_json"]).parent
    targeted = _run_pytest(TARGETED_TESTS)
    env_isaaclab_guard = {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }
    tfds_env = _check_tfds_env(config)
    inventory = inventory_from_config(DEFAULT_CONFIG)
    download = download_from_config(DEFAULT_CONFIG)
    schema = inspect_from_config(DEFAULT_CONFIG)
    manifest = build_manifest_from_config(DEFAULT_CONFIG)
    windows = build_windows_from_config(DEFAULT_CONFIG)
    evaluation = evaluate_tfds_mini_shard(DEFAULT_CONFIG)
    pt_outputs = sorted(str(path) for path in run_dir.glob("**/*.pt"))
    media_outputs = sorted(
        str(path)
        for pattern in ("**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.mp4", "**/*.avi", "**/*.mov", "**/*.mkv")
        for path in run_dir.glob(pattern)
    )
    max_bytes = int(config["download_policy"]["max_download_bytes"])
    safety_pass = bool(
        targeted["passed"]
        and evaluation["safety_gate_pass"]
        and evaluation["download_bytes"] <= max_bytes
        and not evaluation["raw_zip_downloaded"]
        and not evaluation["full_tfds_downloaded"]
        and not evaluation["training_performed"]
        and not evaluation["token_extraction_performed"]
        and not evaluation["importance_generation_performed"]
        and env_isaaclab_guard == {"tensorflow": False, "tensorflow_datasets": False}
        and not pt_outputs
        and not media_outputs
    )
    print("# Step23 Smoke Summary")
    print(
        json.dumps(
            {
                "targeted_tests": targeted,
                "env_isaaclab_guard": env_isaaclab_guard,
                "tfds_env": tfds_env,
                "num_train_shards_found": inventory.get("num_train_shards_found"),
                "download": {
                    "safe_stop": download.get("safe_stop"),
                    "download_performed": download.get("download_performed"),
                    "num_shards_downloaded": download.get("num_shards_downloaded"),
                    "download_bytes": download.get("download_bytes"),
                },
                "schema": {
                    "safe_stop": schema.get("safe_stop"),
                    "num_episodes_scanned": schema.get("num_episodes_scanned"),
                    "can_build_16_4_4_windows": schema.get("can_build_16_4_4_windows"),
                    "reason": schema.get("reason"),
                },
                "manifest": manifest,
                "windows": {
                    "real_tfds_validated": windows.get("real_tfds_validated"),
                    "num_windows": windows.get("num_windows"),
                    "reason": windows.get("reason"),
                },
                "pt_outputs": pt_outputs,
                "media_outputs": media_outputs,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"BRIDGEDATA_V2_TFDS_MINI_VALIDATED = {str(evaluation['real_tfds_validated']).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_MINI_SAFE_STOP = {str(evaluation['safe_stop']).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_MINI_SAFETY_GATE_PASS = {str(safety_pass).lower()}")
    if not safety_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
