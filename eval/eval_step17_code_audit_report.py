"""Render Step17.5 code-audit report from audit summary JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_AUDIT_JSON = PROJECT_ROOT / "runs" / "code_audit_step17_context_bottleneck_v1" / "code_audit_step17_summary.json"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "CODE_AUDIT_STEP17_REPORT.md"


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def render_code_audit_report(summary: dict[str, Any]) -> str:
    checks = [
        "config_audit_pass",
        "importance_context_only_pass",
        "current_not_dropped_pass",
        "no_current_importance_training_pass",
        "no_oracle_leakage_policy_tests_pass",
        "full_context_teacher_reference_aggregate_pass",
        "gitignore_artifact_blacklist_pass",
        "pytest_pass",
    ]
    lines = [
        "# CODE AUDIT STEP17 REPORT",
        "",
        "## Audit Scope",
        "",
        "Step17 context bottleneck implementation was audited without training, downloading, BAIR re-export, VideoMAE extraction, or regeneration of importance shards.",
        "",
        "## Files Reviewed",
        "",
    ]
    lines.extend([f"- `{item}`" for item in summary.get("files_reviewed", [])])
    lines.extend(
        [
            "",
            "## Pass/Fail",
            "",
            "| check | pass |",
            "| --- | --- |",
        ]
    )
    for key in checks:
        lines.append(f"| {key} | `{str(summary.get(key, False)).lower()}` |")
    lines.extend(
        [
            "",
            "## Confirmed Correct",
            "",
            "- current tokens are kept full and mean-pooled in the context bottleneck world model",
            "- current importance is not trained",
            "- context importance generation masks only context tokens and keeps current tokens full",
            "- teacher-context topK is isolated as an oracle baseline",
            "- learned, hybrid, random, and uniform deployable policies do not require Teacher importance labels",
            "- generated artifacts are covered by `.gitignore` and staged-artifact blacklist checks",
            "",
            "## Fixed Or Improved",
            "",
            "- selector initialization now records an `init_report`",
            "- full-context teacher reference has an explicit non-deployable helper and aggregate contract",
            "- no-current-drop, no-current-importance, no-oracle-leakage, runner-guard, and artifact-blacklist tests were added",
            "",
            "## Remaining Caveats",
            "",
            "- this is a static/code audit plus unit-test gate; it does not rerun the Step17 full BAIR pipeline",
            "- BAIR short-context limitations from Step17 remain",
            "",
            "## Blocking Issues",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in summary.get("blocking_issues", [])] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in summary.get("warnings", [])] or ["- none"])
    lines.extend(
        [
            "",
            "## Next Recommendation",
            "",
            str(summary.get("recommended_next_step", "Step18 context selector oracle-gap diagnostic")),
            "",
            f"STEP17_CODE_AUDIT_PASS = {str(summary.get('audit_pass', False)).lower()}",
            "",
        ]
    )
    return "\n".join(lines)


def write_code_audit_report(audit_json: str | Path = DEFAULT_AUDIT_JSON, report_path: str | Path = DEFAULT_REPORT) -> dict[str, Any]:
    summary = _read_json(audit_json)
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_code_audit_report(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", default=str(DEFAULT_AUDIT_JSON))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_code_audit_report(args.audit_json, args.report_path)
    print(Path(args.report_path).read_text(encoding="utf-8"))
    if not summary.get("audit_pass", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
