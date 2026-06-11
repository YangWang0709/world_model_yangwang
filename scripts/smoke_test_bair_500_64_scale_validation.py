"""Validation wrapper for Step 14 BAIR 500/64 scale validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bair_500_64_scale_validation import (
    load_yaml,
    render_scale_validation_markdown,
    write_scale_validation_summary,
)
from scripts.run_bair_500_64_scale_validation import DEFAULT_CONFIG, run_scale_validation


REPORT_PATH = PROJECT_ROOT / "docs" / "BAIR_500_64_SCALE_VALIDATION_REPORT.md"
STEP_DOC_PATH = PROJECT_ROOT / "docs" / "STEP14_BAIR_500_64_SCALE_VALIDATION.md"


def _run_pytest() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
    return {
        "command": f"{sys.executable} -m pytest tests -q",
        "returncode": int(result.returncode),
        "passed": result.returncode == 0,
        "output_tail": "\n".join(output.strip().splitlines()[-20:]),
    }


def _required_paths(config: dict[str, Any]) -> list[Path]:
    run_root = Path(config["output"]["run_root"])
    return [
        Path(config["dataset"]["subset_dir"]) / "train" / "metadata.jsonl",
        Path(config["dataset"]["subset_dir"]) / "test" / "metadata.jsonl",
        Path(config["tokens"]["output_root"]) / "extraction_summary.json",
        run_root / config["teacher"]["run_name"] / "summary.json",
        run_root / config["teacher"]["run_name"] / "eval_summary.json",
        Path(config["importance"]["output_root"]) / "importance_summary.json",
        Path(config["importance"]["output_root"]) / "eval_importance_summary.json",
        run_root / config["selector"]["run_name"] / "summary.json",
        run_root / config["student_world_model"]["run_name"] / "summary.json",
        run_root / config["student_world_model"]["run_name"] / "eval_summary.json",
        run_root / config["student_world_model"]["run_name"] / "teacher_student_gap_summary.json",
        run_root / config["baseline"]["run_name"] / "baseline_summary.json",
        Path(config["output"]["summary_json"]),
        Path(config["output"]["summary_md"]),
    ]


def _assert_required_outputs(config: dict[str, Any]) -> None:
    missing = [str(path) for path in _required_paths(config) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Step 14 outputs: {missing}")


def write_validation_report(summary: dict[str, Any], pytest_result: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# BAIR 500/64 Scale Validation Report",
        "",
        "Command: `python scripts/smoke_test_bair_500_64_scale_validation.py`",
        "",
        "## Output Dirs",
        "",
        f"- subset: `{summary.get('subset_path')}`",
        f"- tokens: `{summary.get('token_output_root')}`",
        f"- importance: `{summary.get('importance_output_root')}`",
        f"- Teacher run: `{summary.get('teacher_summary', {}).get('run_dir')}`",
        f"- Student run: `{summary.get('student_summary', {}).get('run_dir')}`",
        "",
        "## Unified Summary",
        "",
        render_scale_validation_markdown(summary),
        "",
        "## Pytest",
        "",
        "```text",
        pytest_result.get("output_tail", ""),
        "```",
        "",
        f"BAIR_500_64_SCALE_VALIDATION_PASS = {str(summary.get('sanity_gate_pass')).lower()}",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def write_step_doc(summary: dict[str, Any], pytest_result: dict[str, Any]) -> None:
    lines = [
        "# STEP14 BAIR 500/64 Scale Validation",
        "",
        "## 1. Goal",
        "",
        "Scale the BAIR Robot Pushing small validation from 100/16 smoke artifacts to 500 train and 64 test clips while keeping the Step 13 selector choice fixed.",
        "",
        "## 2. Why Stop Smoke Tuning",
        "",
        "Step 13 selected `weighted_mse_alpha2` by downstream MSE on the 100/16 setting. Step 14 validates that choice at a larger scale instead of adding more small-sample loss variants.",
        "",
        "## 3. Cloud Server Decision",
        "",
        "No cloud server was required for this stage. The local RTX 5080 host is enough unless CUDA OOM persists after reducing batch size or token chunk size, in which case a 4090 / 48GB host becomes a reasonable next option.",
        "",
        "## 4. Input",
        "",
        f"- BAIR TFDS dir: `{summary.get('subset_path')}` source export from existing TFDS",
        "- VideoMAE checkpoint: local `model_cache/videomae-base-finetuned-kinetics-local`",
        f"- train/test samples: `{summary.get('train_samples')}` / `{summary.get('test_samples')}`",
        f"- token shape: `{summary.get('token_shape')}`",
        f"- topK: `{summary.get('topk')}`",
        "",
        "## 5. Pipeline",
        "",
        "BAIR subset export -> VideoMAE tokens -> Teacher -> predictive importance -> weighted_mse_alpha2 selector -> StudentWorldModel -> baselines.",
        "",
        "## 6. Configs",
        "",
        "- `configs/bair_500_64_scale_validation.yaml`",
        "- `configs/export_bair_subset_500_64.yaml`",
        "- `configs/token_extraction_bair_videomae_500_64.yaml`",
        "- `configs/train_teacher_bair_videomae_500_64.yaml`",
        "- `configs/generate_importance_bair_videomae_500_64.yaml`",
        "- `configs/train_selector_bair_videomae_500_64_weighted_mse.yaml`",
        "- `configs/train_student_world_model_bair_videomae_500_64_weighted_mse.yaml`",
        "- `configs/baseline_comparison_bair_videomae_500_64.yaml`",
        "",
        "## 7. Results",
        "",
        f"- Teacher MSE: `{summary.get('teacher_eval_mse')}`",
        f"- Student MSE: `{summary.get('student_future_mse')}`",
        f"- Student/Teacher ratio: `{summary.get('student_teacher_ratio')}`",
        f"- learned vs random mean MSE delta: `{summary.get('learned_vs_random_mse_delta')}`",
        f"- learned vs uniform MSE delta: `{summary.get('learned_vs_uniform_mse_delta')}`",
        f"- learned vs teacher_importance_topk MSE delta: `{summary.get('learned_vs_teacher_importance_topk_mse_delta')}`",
        "",
        "See `docs/BAIR_500_64_SCALE_VALIDATION_REPORT.md` for the full table.",
        "",
        "## 8. Sanity Gate",
        "",
        "```json",
        json.dumps(summary.get("sanity_gate", {}), indent=2),
        "```",
        "",
        "## 9. Resource Usage",
        "",
        "```json",
        json.dumps(summary.get("resource_summary", {}), indent=2),
        "```",
        "",
        "## 10. Pytest",
        "",
        f"- command: `{pytest_result.get('command')}`",
        f"- passed: `{str(pytest_result.get('passed')).lower()}`",
        "",
        "## 11. Git Commit",
        "",
        "This document records the Step 14 code/report state. Commit hash is filled in the final local summary after push.",
        "",
        "## 12. What Was Not Done",
        "",
        "- no new model download",
        "- no new dataset download",
        "- no BAIR re-download",
        "- no VideoMAE training",
        "- no Step 11E/11F/12/13 overwrite",
        "- no VLM grounding",
        "- no RL",
        "- no action-conditioned world model",
        "- no model weight committed",
        "- no BAIR data committed",
        "- no token shard committed",
        "- no importance shard committed",
        "- no checkpoint committed",
        "- no password/token saved",
        "- no PR created",
        "",
        "## 13. Next Step Recommendation",
        "",
        "If the learned selector is stable against random and uniform, Step 15 can move to 1000/128 or multi-seed. If it only beats random, improve the compressor or StudentWorldModel before scaling further. If it is unstable, inspect Teacher importance quality.",
        "",
    ]
    STEP_DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    run_scale_validation(DEFAULT_CONFIG)
    _assert_required_outputs(config)
    pytest_result = _run_pytest()
    summary = write_scale_validation_summary(config, pytest_result=pytest_result)
    write_validation_report(summary, pytest_result)
    write_step_doc(summary, pytest_result)
    print(REPORT_PATH.read_text(encoding="utf-8"))
    print(f"BAIR_500_64_SCALE_VALIDATION_PASS = {str(summary.get('sanity_gate_pass')).lower()}")
    if not summary.get("sanity_gate_pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
