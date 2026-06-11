"""Validation wrapper for Step17 BAIR context bottleneck pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_bair_context_bottleneck_validation import load_yaml, render_context_bottleneck_markdown, write_context_bottleneck_summaries
from training.run_bair_context_bottleneck_validation import DEFAULT_CONFIG, run_bair_context_bottleneck_validation


REPORT_PATH = PROJECT_ROOT / "docs" / "BAIR_CONTEXT_BOTTLENECK_VALIDATION_REPORT.md"
STEP_DOC_PATH = PROJECT_ROOT / "docs" / "STEP17_BAIR_CONTEXT_BOTTLENECK_WORLD_MODEL.md"


def _run_pytest() -> dict[str, Any]:
    result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=str(PROJECT_ROOT), check=False, capture_output=True, text=True)
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
    return {
        "command": f"{sys.executable} -m pytest tests -q",
        "returncode": int(result.returncode),
        "passed": result.returncode == 0,
        "output_tail": "\n".join(output.strip().splitlines()[-30:]),
    }


def _env_guard() -> dict[str, bool]:
    import importlib.util

    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.8f}"
    except (TypeError, ValueError):
        return "n/a"


def write_validation_report(summary: dict[str, Any], pytest_result: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BAIR Context Bottleneck Validation Report",
        "",
        "Command: `python scripts/smoke_test_bair_context_bottleneck_validation.py`",
        "",
        "## Outputs",
        "",
        f"- context windows: `{summary.get('context_window_summary', {}).get('output_root')}`",
        f"- context tokens: `{summary.get('token_extraction_summary', {}).get('output_root')}`",
        f"- context importance: `{summary.get('context_importance_summary', {}).get('output_root')}`",
        f"- run dir: `{summary.get('run_dir')}`",
        "",
        "## Unified Summary",
        "",
        render_context_bottleneck_markdown(summary),
        "",
        "## Pytest",
        "",
        "```text",
        pytest_result.get("output_tail", ""),
        "```",
        "",
        f"BAIR_CONTEXT_BOTTLENECK_VALIDATION_PASS = {str(summary.get('sanity_gate_pass')).lower()}",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_step_doc(summary: dict[str, Any], pytest_result: dict[str, Any]) -> None:
    lines = [
        "# STEP17 BAIR Context Bottleneck World Model",
        "",
        "## 1. Goal",
        "",
        "Validate the shift from state/current bottlenecking to historical context bottlenecking on BAIR 1000/128.",
        "",
        "## 2. Motivation",
        "",
        "The current observation is kept complete and only summarized. Predictive importance selection is applied to historical context tokens only.",
        "",
        "## 3. Key Design Decision",
        "",
        "- current_tokens are not topK-dropped",
        "- current importance is not trained in this step",
        "- only context importance is trained",
        "- UnifiedPredictiveImportanceSelector is used in context mode",
        "- state_legacy mode is retained only for compatibility",
        "",
        "## 4. Cloud Server Decision",
        "",
        f"Cloud server required: `{str(summary.get('cloud_required')).lower()}`. Local execution is sufficient unless OOM persists after reducing batch size or token_chunk_size.",
        "",
        "## 5. Dataset",
        "",
        f"- train/test: `{summary.get('train_samples')}` / `{summary.get('test_samples')}`",
        f"- context/current/future: `{summary.get('context_len')}` / `{summary.get('current_len')}` / `{summary.get('future_len')}` frames",
        "- action/endeffector are saved as metadata and are not model inputs",
        "",
        "## 6. Token Extraction",
        "",
        f"- context_topK: `{summary.get('context_topK')}`",
        f"- context retention ratio: `{summary.get('context_retention_ratio')}`",
        "- VideoMAE local checkpoint only; no fallback",
        "",
        "## 7. Context Teacher",
        "",
        f"- eval MSE: `{_fmt(summary.get('context_teacher_mse'))}`",
        "",
        "## 8. Context Importance",
        "",
        "I_i = L(Teacher(context_without_i, current_full), future) - L(Teacher(context_full, current_full), future)",
        "",
        "## 9. Unified Predictive Importance Selector",
        "",
        f"- importance MSE: `{_fmt(summary.get('unified_selector_summary', {}).get('importance_mse'))}`",
        f"- pearson corr: `{_fmt(summary.get('unified_selector_summary', {}).get('pearson_corr'))}`",
        f"- target topK overlap: `{_fmt(summary.get('unified_selector_summary', {}).get('target_topK_overlap'))}`",
        "",
        "## 10. Context Bottleneck World Model",
        "",
        f"- learned_context MSE: `{_fmt(summary.get('learned_context_mse'))}`",
        f"- current_only MSE: `{_fmt(summary.get('current_only_mse'))}`",
        f"- context gain: `{_fmt(summary.get('context_gain_over_current_only'))}`",
        "",
        "## 11. Baselines",
        "",
        render_context_bottleneck_markdown(summary),
        "",
        "## 12. BAIR Context Limitation",
        "",
        str(summary.get("bair_context_limitation")),
        "",
        "## 13. Sanity Gate",
        "",
        "```json",
        json.dumps(summary.get("sanity_gate", {}), indent=2),
        "```",
        "",
        "## 14. Pytest",
        "",
        f"- command: `{pytest_result.get('command')}`",
        f"- passed: `{str(pytest_result.get('passed')).lower()}`",
        "",
        "## 15. What Was Not Done",
        "",
        "- no new model download",
        "- no new dataset download",
        "- no BAIR re-download",
        "- no VideoMAE training",
        "- no current importance training",
        "- no current token topK dropping",
        "- no Step11-16 overwrite",
        "- no VLM grounding",
        "- no RL",
        "- no action-conditioned world model",
        "- action/endeffector saved but not used as input",
        "- no model weight committed",
        "- no BAIR/context data committed",
        "- no token/importance shard committed",
        "- no checkpoint committed",
        "- no password/token saved",
        "- no PR created",
        "",
    ]
    STEP_DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    runner_summary = run_bair_context_bottleneck_validation(DEFAULT_CONFIG)
    pytest_result = _run_pytest()
    env_guard = _env_guard()
    summary = write_context_bottleneck_summaries(
        config=config,
        resource_summary=runner_summary.get("resource_summary", {}),
        pytest_result=pytest_result,
        env_guard=env_guard,
    )
    write_validation_report(summary, pytest_result)
    write_step_doc(summary, pytest_result)
    print(REPORT_PATH.read_text(encoding="utf-8"))
    print(f"BAIR_CONTEXT_BOTTLENECK_VALIDATION_PASS = {str(summary.get('sanity_gate_pass')).lower()}")
    if not summary.get("sanity_gate_pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
