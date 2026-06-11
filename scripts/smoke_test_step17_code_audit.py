"""Smoke gate for Step17.5 code-audit patch."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_step17_code_audit_report import write_code_audit_report
from scripts.audit_step17_context_bottleneck import DEFAULT_CONFIG, REQUIRED_TESTS, run_step17_code_audit


STEP_DOC_PATH = PROJECT_ROOT / "docs" / "STEP17_5_CODE_AUDIT_PATCH.md"
REPORT_PATH = PROJECT_ROOT / "docs" / "CODE_AUDIT_STEP17_REPORT.md"


def _run_pytest(paths: list[str] | None = None) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest"]
    cmd.extend(paths or ["tests"])
    cmd.append("-q")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False, capture_output=True, text=True)
    output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
    return {
        "command": " ".join(cmd),
        "returncode": int(result.returncode),
        "passed": result.returncode == 0,
        "output_tail": "\n".join(output.strip().splitlines()[-50:]),
    }


def _env_guard() -> dict[str, bool]:
    import importlib.util

    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def write_step_doc(summary: dict[str, Any], targeted: dict[str, Any], full: dict[str, Any]) -> None:
    lines = [
        "# STEP17.5 Code Audit Patch",
        "",
        "## 1. Goal",
        "",
        "This is not a new experiment. It strengthens code trust before context-selector oracle-gap work.",
        "",
        "## 2. Scope",
        "",
        "- current tokens are not dropped",
        "- current importance is not trained",
        "- context-only occlusion",
        "- unified selector context mode",
        "- oracle baseline isolation",
        "- full_context_teacher_reference aggregation",
        "- artifact blacklist",
        "",
        "## 3. Findings Before Patch",
        "",
        "- core Step17 design was mostly correct",
        "- tests were shallow around no-current-drop and oracle leakage",
        "- selector initialization did not expose a detailed report",
        "- full_context_teacher_reference needed an explicit aggregate contract",
        "",
        "## 4. Changes Made",
        "",
        "- added `init_report` for context selector initialization",
        "- added full-context teacher reference helper and aggregate handling",
        "- added Step17.5 audit scripts, smoke gate, and report generation",
        "- added focused tests for current-token, current-importance, occlusion, oracle leakage, protected paths, and artifacts",
        "",
        "## 5. Audit Results",
        "",
        f"- audit pass: `{str(summary.get('audit_pass')).lower()}`",
        f"- blocking issues: `{len(summary.get('blocking_issues', []))}`",
        f"- warnings: `{len(summary.get('warnings', []))}`",
        "",
        "## 6. Pytest",
        "",
        f"- targeted audit tests: `{targeted.get('output_tail', '').splitlines()[-1] if targeted.get('output_tail') else ''}`",
        f"- full pytest: `{full.get('output_tail', '').splitlines()[-1] if full.get('output_tail') else ''}`",
        "",
        "## 7. What Was Not Done",
        "",
        "- no training",
        "- no model download",
        "- no dataset download",
        "- no BAIR re-download",
        "- no VideoMAE extraction",
        "- no current importance training",
        "- no current token topK drop",
        "- no VLM/RL/action-conditioned model",
        "- no data/checkpoint/runs committed",
        "- no PR created",
        "",
        "## 8. Next Step",
        "",
        "Proceed to Step18 context selector oracle-gap diagnostic if audit passes.",
        "",
        f"STEP17_CODE_AUDIT_PASS = {str(summary.get('audit_pass')).lower()}",
        "",
    ]
    STEP_DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    targeted = _run_pytest(REQUIRED_TESTS)
    full = _run_pytest(["tests"])
    env_guard = _env_guard()
    full_with_env = {**full, "env_guard": env_guard}
    summary = run_step17_code_audit(DEFAULT_CONFIG, pytest_result=full_with_env)
    write_code_audit_report()
    write_step_doc(summary, targeted, full)
    passed = bool(targeted["passed"] and full["passed"] and summary["audit_pass"] and not env_guard["tensorflow"] and not env_guard["tensorflow_datasets"])
    print(REPORT_PATH.read_text(encoding="utf-8"))
    print("## Targeted Pytest")
    print("```text")
    print(targeted["output_tail"])
    print("```")
    print("## Full Pytest")
    print("```text")
    print(full["output_tail"])
    print("```")
    print(f"STEP17_CODE_AUDIT_PASS = {str(passed).lower()}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
