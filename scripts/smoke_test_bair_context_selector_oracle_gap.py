"""Validation wrapper for Step18 BAIR context-selector oracle-gap diagnostic."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bair_context_selector_oracle_gap import (
    render_context_selector_oracle_gap_markdown,
    write_context_selector_oracle_gap_summary,
)
from training.context_selector_oracle_gap_trainer import (
    DEFAULT_CONFIG,
    is_step18_owned_output_path,
    load_yaml,
    run_bair_context_selector_oracle_gap,
)


REPORT_PATH = PROJECT_ROOT / "docs" / "BAIR_CONTEXT_SELECTOR_ORACLE_GAP_REPORT.md"
STEP_DOC_PATH = PROJECT_ROOT / "docs" / "STEP18_BAIR_CONTEXT_SELECTOR_ORACLE_GAP.md"


def _run_pytest() -> dict[str, Any]:
    result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=str(PROJECT_ROOT), check=False, capture_output=True, text=True)
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
    return {
        "command": f"{sys.executable} -m pytest tests -q",
        "returncode": int(result.returncode),
        "passed": result.returncode == 0,
        "output_tail": "\n".join(output.strip().splitlines()[-40:]),
    }


def _env_guard() -> dict[str, bool]:
    import importlib.util

    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def _safe_clean_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        if not is_step18_owned_output_path(run_dir):
            raise ValueError(f"Refusing to delete non-Step18 run dir: {run_dir}")
        shutil.rmtree(run_dir)


def _check_required_outputs(run_dir: Path) -> list[str]:
    required = [
        run_dir / "label_diagnostic" / "context_importance_diagnostic.json",
        run_dir / "label_diagnostic" / "context_importance_diagnostic.md",
        run_dir / "selector_phase_a_summary.json",
        run_dir / "downstream_phase_a_summary.json",
        run_dir / "selector_phase_b_summary.json",
        run_dir / "downstream_phase_b_summary.json",
        run_dir / "context_selector_oracle_gap_summary.json",
        run_dir / "context_selector_oracle_gap_summary.md",
        REPORT_PATH,
        STEP_DOC_PATH,
    ]
    return [str(path) for path in required if not path.exists()]


def write_validation_report(summary: dict[str, Any], pytest_result: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BAIR Context Selector Oracle-Gap Report",
        "",
        "Command: `python scripts/smoke_test_bair_context_selector_oracle_gap.py`",
        "",
        "## Input Step17 Paths",
        "",
        "- context tokens: `/home/ubuntu22/tgpawb_world_model/data/context_token_shards/bair_context_videomae_1000_128`",
        "- context importance: `/home/ubuntu22/tgpawb_world_model/data/context_importance_shards/bair_context_teacher_1000_128`",
        "- ContextTeacher checkpoint: `/home/ubuntu22/tgpawb_world_model/runs/context_teacher_bair_1000_128_v1/checkpoints/context_teacher_step_001200.pt`",
        "- Step17 baseline aggregate: `/home/ubuntu22/tgpawb_world_model/runs/context_bottleneck_baseline_bair_1000_128_v1/baseline_aggregate.json`",
        "",
        render_context_selector_oracle_gap_markdown(summary),
        "",
        "## Pytest",
        "",
        "```text",
        pytest_result.get("output_tail", ""),
        "```",
        "",
        f"BAIR_CONTEXT_SELECTOR_ORACLE_GAP_PASS = {str(summary.get('sanity_gate_pass')).lower()}",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_step_doc(summary: dict[str, Any], pytest_result: dict[str, Any]) -> None:
    lines = [
        "# STEP18 BAIR Context Selector Oracle-Gap Diagnostic",
        "",
        "## 1. Goal",
        "",
        "Diagnose why `teacher_context_importance_topK` is strong while the learned context selector did not beat random in Step17.",
        "",
        "## 2. Motivation",
        "",
        "Step17 proved the context bottleneck pipeline works and that historical context contains useful signal. The remaining bottleneck is the learned selector's oracle gap.",
        "",
        "## 3. Key Constraint",
        "",
        "- current tokens are never topK-dropped",
        "- current importance is not trained",
        "- only context importance is trained",
        "- action and endeffector metadata are not model inputs",
        "",
        "## 4. Cloud Server Decision",
        "",
        "No cloud server was required. The local Ubuntu RTX 5080 host stayed within disk/RAM/GPU limits.",
        "",
        "## 5. Inputs",
        "",
        "- Step17 context/current/future tokens were reused",
        "- Step17 context importance labels were reused",
        "- Step17 ContextTeacher checkpoint was reused",
        "- Step17 baseline aggregate was reused",
        "",
        "## 6. Label Diagnostic",
        "",
        f"- positive ratio: `{summary['label_diagnostic']['positive_importance_ratio']:.6f}`",
        f"- top32 mass ratio: `{summary['label_diagnostic']['topk_concentration'].get('top32_mass_ratio', 0.0):.8f}`",
        f"- oracle vs random importance gap: `{summary['label_diagnostic']['oracle_random_gap']['oracle_vs_random_importance_gap']:.8f}`",
        "",
        "## 7. Selector Variants",
        "",
        "- weighted_mse_alpha2",
        "- weighted_mse_alpha5",
        "- topK BCE",
        "- pairwise ranking",
        "- hybrid weighted MSE + ranking + BCE",
        "- no current conditioning",
        "- no temporal position",
        "- temporal-block balanced topK",
        "",
        "## 8. Results",
        "",
        render_context_selector_oracle_gap_markdown(summary),
        "",
        "## 9. Pytest",
        "",
        f"- command: `{pytest_result.get('command')}`",
        f"- passed: `{str(pytest_result.get('passed')).lower()}`",
        "",
        "## 10. What Was Not Done",
        "",
        "- no new model download",
        "- no new dataset download",
        "- no BAIR re-download",
        "- no VideoMAE token re-extraction",
        "- no VideoMAE training",
        "- no ContextTeacher retraining",
        "- no context importance regeneration",
        "- no current importance training",
        "- no current token topK dropping",
        "- no Step11-17 overwrite",
        "- no VLM",
        "- no RL",
        "- no action-conditioned world model",
        "- no model weight committed",
        "- no BAIR data committed",
        "- no context windows committed",
        "- no token shard committed",
        "- no importance shard committed",
        "- no checkpoint committed",
        "- no password/token saved",
        "- no PR created",
        "",
    ]
    STEP_DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    _safe_clean_run_dir(run_dir)
    run_bair_context_selector_oracle_gap(DEFAULT_CONFIG)
    pytest_result = _run_pytest()
    env_guard = _env_guard()
    summary = write_context_selector_oracle_gap_summary(
        run_dir=run_dir,
        step17_summary_path=config["step17_reference"]["context_bottleneck_summary"],
        step17_baseline_path=config["step17_reference"]["baseline_aggregate"],
        pytest_result=pytest_result,
        env_guard=env_guard,
    )
    write_validation_report(summary, pytest_result)
    write_step_doc(summary, pytest_result)
    missing = _check_required_outputs(run_dir)
    if missing:
        raise RuntimeError(f"Missing Step18 output(s): {missing}")
    print(REPORT_PATH.read_text(encoding="utf-8"))
    print(f"BAIR_CONTEXT_SELECTOR_ORACLE_GAP_PASS = {str(summary.get('sanity_gate_pass')).lower()}")
    if not summary.get("sanity_gate_pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
