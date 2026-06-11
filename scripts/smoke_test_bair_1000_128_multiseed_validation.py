"""Validation wrapper for Step 15 BAIR 1000/128 multi-seed validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bair_1000_128_multiseed_validation import (
    load_yaml,
    render_multiseed_validation_markdown,
    write_multiseed_validation_summary,
)
from scripts.run_bair_1000_128_multiseed_validation import DEFAULT_CONFIG, run_multiseed_validation


REPORT_PATH = PROJECT_ROOT / "docs" / "BAIR_1000_128_MULTI_SEED_VALIDATION_REPORT.md"
STEP_DOC_PATH = PROJECT_ROOT / "docs" / "STEP15_BAIR_1000_128_MULTI_SEED_VALIDATION.md"


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
    selector_root = run_root / config["selector"]["run_root_name"]
    student_root = run_root / config["student_world_model"]["run_root_name"]
    baseline_root = run_root / config["baseline"]["run_name"]
    return [
        Path(config["dataset"]["subset_dir"]) / "train" / "metadata.jsonl",
        Path(config["dataset"]["subset_dir"]) / "test" / "metadata.jsonl",
        Path(config["tokens"]["output_root"]) / "extraction_summary.json",
        run_root / config["teacher"]["run_name"] / "summary.json",
        run_root / config["teacher"]["run_name"] / "eval_summary.json",
        Path(config["importance"]["output_root"]) / "importance_summary.json",
        Path(config["importance"]["output_root"]) / "eval_importance_summary.json",
        selector_root / "selector_multiseed_summary.json",
        student_root / "student_world_model_multiseed_summary.json",
        baseline_root / "baseline_summary.json",
        baseline_root / "baseline_summary.csv",
        baseline_root / "baseline_summary.md",
        baseline_root / "baseline_multiseed_aggregate.json",
        baseline_root / "baseline_multiseed_aggregate.csv",
        baseline_root / "baseline_multiseed_aggregate.md",
        Path(config["output"]["summary_json"]),
        Path(config["output"]["summary_md"]),
    ]


def _assert_required_outputs(config: dict[str, Any]) -> None:
    missing = [str(path) for path in _required_paths(config) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Step 15 outputs: {missing}")


def write_validation_report(summary: dict[str, Any], pytest_result: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# BAIR 1000/128 Multi-Seed Validation Report",
        "",
        "Command: `python scripts/smoke_test_bair_1000_128_multiseed_validation.py`",
        "",
        "## Output Dirs",
        "",
        f"- subset: `{summary.get('subset_path')}`",
        f"- tokens: `{summary.get('token_output_root')}`",
        f"- importance: `{summary.get('importance_output_root')}`",
        f"- Teacher run: `{summary.get('teacher_summary', {}).get('run_dir')}`",
        f"- selector multi-seed run: `runs/student_selector_bair_videomae_1000_128_weighted_mse_multiseed_v1`",
        f"- StudentWorldModel multi-seed run: `runs/student_world_model_bair_videomae_1000_128_weighted_mse_multiseed_v1`",
        f"- baseline multi-seed run: `runs/baseline_comparison_bair_videomae_1000_128_multiseed_v1`",
        "",
        "## Unified Summary",
        "",
        render_multiseed_validation_markdown(summary),
        "",
        "## Pytest",
        "",
        "```text",
        pytest_result.get("output_tail", ""),
        "```",
        "",
        f"BAIR_1000_128_MULTI_SEED_VALIDATION_PASS = {str(summary.get('sanity_gate_pass')).lower()}",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def write_step_doc(summary: dict[str, Any], pytest_result: dict[str, Any]) -> None:
    lines = [
        "# STEP15 BAIR 1000/128 Multi-Seed Validation",
        "",
        "## 1. Goal",
        "",
        "Scale the BAIR Robot Pushing small validation from Step 14's 500/64 setting to 1000 train clips and 128 test clips, then evaluate stability across multiple seeds.",
        "",
        "## 2. Why Multi-Seed Now",
        "",
        "Step 14 fixed the selector choice to `weighted_mse_alpha2` and showed it beating random and uniform on 500/64. Step 15 tests whether that result is stable rather than tuning more small-sample variants.",
        "",
        "## 3. Cloud Server Decision",
        "",
        "No cloud server is required when the local RTX 5080 run stays under the disk/RAM/GPU limits. A 4090 / 48GB host is only a recommendation if CUDA OOM persists after reducing batch size or token chunk size.",
        "",
        "## 4. Input",
        "",
        f"- BAIR subset path: `{summary.get('subset_path')}`",
        "- VideoMAE checkpoint: local `model_cache/videomae-base-finetuned-kinetics-local`",
        f"- train/test samples: `{summary.get('train_samples')}` / `{summary.get('test_samples')}`",
        f"- token shape: `{summary.get('token_shape')}`",
        f"- topK: `{summary.get('topk')}`",
        f"- seeds: `{summary.get('selector_seeds')}`",
        "",
        "## 5. Pipeline",
        "",
        "BAIR subset export -> VideoMAE tokens -> Teacher -> predictive importance -> weighted_mse_alpha2 selector multi-seed -> StudentWorldModel multi-seed -> baseline comparison multi-seed.",
        "",
        "## 6. Configs",
        "",
        "- `configs/bair_1000_128_multiseed_validation.yaml`",
        "- `configs/export_bair_subset_1000_128.yaml`",
        "- `configs/token_extraction_bair_videomae_1000_128.yaml`",
        "- `configs/train_teacher_bair_videomae_1000_128.yaml`",
        "- `configs/generate_importance_bair_videomae_1000_128.yaml`",
        "- `configs/train_selector_bair_videomae_1000_128_weighted_mse.yaml`",
        "- `configs/train_student_world_model_bair_videomae_1000_128_weighted_mse.yaml`",
        "- `configs/baseline_comparison_bair_videomae_1000_128_multiseed.yaml`",
        "",
        "## 7. Results",
        "",
        f"- Teacher MSE: `{summary.get('teacher_eval_mse')}`",
        f"- learned MSE mean/std: `{summary.get('baseline_learned_mse_mean')}` / `{summary.get('baseline_learned_mse_std')}`",
        f"- random MSE mean/std: `{summary.get('baseline_random_mse_mean')}` / `{summary.get('baseline_random_mse_std')}`",
        f"- uniform MSE: `{summary.get('baseline_uniform_mse')}`",
        f"- teacher_importance_topk MSE: `{summary.get('baseline_teacher_importance_topk_mse')}`",
        f"- learned vs random mean MSE delta: `{summary.get('learned_vs_random_mse_delta')}`",
        f"- learned vs uniform MSE delta: `{summary.get('learned_vs_uniform_mse_delta')}`",
        f"- learned vs teacher_importance_topk MSE delta: `{summary.get('learned_vs_teacher_importance_topk_mse_delta')}`",
        "",
        "See `docs/BAIR_1000_128_MULTI_SEED_VALIDATION_REPORT.md` for the full tables.",
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
        "This document records the Step 15 code/report state. Commit hash is filled in the final local summary after push.",
        "",
        "## 12. What Was Not Done",
        "",
        "- no new model download",
        "- no new dataset download",
        "- no BAIR re-download",
        "- no VideoMAE training",
        "- no Step 11E/11F/12/13/14 overwrite",
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
        "If learned weighted_mse_alpha2 remains stable against random and uniform, Step 16 can either scale again or start the next modeling upgrade. If it only beats random, improve the compressor/StudentWorldModel first. If it is unstable, diagnose Teacher importance quality and seed control.",
        "",
    ]
    STEP_DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    run_multiseed_validation(DEFAULT_CONFIG)
    _assert_required_outputs(config)
    pytest_result = _run_pytest()
    summary = write_multiseed_validation_summary(config, pytest_result=pytest_result)
    write_validation_report(summary, pytest_result)
    write_step_doc(summary, pytest_result)
    print(REPORT_PATH.read_text(encoding="utf-8"))
    print(f"BAIR_1000_128_MULTI_SEED_VALIDATION_PASS = {str(summary.get('sanity_gate_pass')).lower()}")
    if not summary.get("sanity_gate_pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
