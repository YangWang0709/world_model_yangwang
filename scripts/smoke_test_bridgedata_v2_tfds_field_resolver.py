"""Smoke gate for Step23.5 BridgeData V2 TFDS/RLDS field resolver."""

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

from eval.eval_bridgedata_v2_tfds_field_resolver import evaluate_field_resolver
from scripts.rebuild_bridgedata_v2_tfds_mini_manifest_resolved import rebuild_resolved_manifest_and_windows

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_field_resolver_step23_5.yaml"
TARGETED_TESTS = [
    "tests/test_bridgedata_v2_rlds_field_resolver.py",
    "tests/test_bridgedata_v2_rlds_field_resolver_manifest.py",
    "tests/test_bridgedata_v2_tfds_field_resolver_report.py",
    "tests/test_bridgedata_v2_tfds_field_resolver_no_download.py",
    "tests/test_bridgedata_v2_tfds_field_resolver_artifact_blacklist.py",
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


def main() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    run_dir = Path(config["output"]["eval_json"]).parent
    targeted = _run_pytest(TARGETED_TESTS)
    env_guard = {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }
    rebuild = rebuild_resolved_manifest_and_windows(DEFAULT_CONFIG)
    evaluation = evaluate_field_resolver(DEFAULT_CONFIG)
    pt_outputs = sorted(str(path) for path in run_dir.glob("**/*.pt"))
    media_outputs = sorted(
        str(path)
        for pattern in ("**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.mp4", "**/*.avi", "**/*.mov", "**/*.mkv")
        for path in run_dir.glob(pattern)
    )
    resolved = rebuild["resolved_fields"]
    safety_gate = bool(
        targeted["passed"]
        and env_guard == {"tensorflow": False, "tensorflow_datasets": False}
        and evaluation["download_performed"] is False
        and evaluation["training_performed"] is False
        and evaluation["token_extraction_performed"] is False
        and evaluation["importance_generation_performed"] is False
        and not pt_outputs
        and not media_outputs
    )
    pass_gate = bool(
        evaluation["pass"]
        and resolved.get("image_field") == "steps/observation/image_0"
        and resolved.get("language_field") == "steps/language_instruction"
        and resolved.get("action_field") == "steps/action"
        and safety_gate
    )
    print("# Step23.5 Smoke Summary")
    print(
        json.dumps(
            {
                "targeted_tests": targeted,
                "env_isaaclab_guard": env_guard,
                "resolved_fields": resolved,
                "evaluation": {
                    "pass": evaluation["pass"],
                    "num_manifest_records": evaluation["num_manifest_records"],
                    "num_windows": evaluation["num_windows"],
                },
                "pt_outputs": pt_outputs,
                "media_outputs": media_outputs,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"BRIDGEDATA_V2_TFDS_FIELD_RESOLVER_PASS = {str(pass_gate).lower()}")
    print(f"BRIDGEDATA_V2_TFDS_RESOLVED_IMAGE_FIELD = {resolved.get('image_field')}")
    print(f"BRIDGEDATA_V2_TFDS_FIELD_RESOLVER_SAFETY_GATE_PASS = {str(safety_gate).lower()}")
    if not pass_gate or not safety_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
