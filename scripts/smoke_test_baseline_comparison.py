"""End-to-end Step 8 smoke test for baseline selection-policy comparison."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.eval_baseline_comparison import evaluate_baseline_comparison
from scripts.smoke_test_student_world_model_training import main as run_step7_smoke
from training.run_baseline_comparison import load_yaml
from training.baseline_student_world_model_trainer import train_baseline_comparison


REPORT_PATH = PROJECT_ROOT / "docs" / "BASELINE_COMPARISON_SMOKE_REPORT.md"
STEP8_DOC_PATH = PROJECT_ROOT / "docs" / "STEP8_BASELINE_COMPARISON.md"


def _project_config(config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    updated["data"] = dict(updated["data"])
    updated["teacher_reference"] = dict(updated["teacher_reference"])
    updated["learned_selector"] = dict(updated["learned_selector"])
    updated["output"] = dict(updated["output"])
    run_dir = PROJECT_ROOT / "runs" / "baseline_comparison_structured_toy_v1"
    updated["data"]["token_shard_dir"] = str(PROJECT_ROOT / "data" / "token_shards" / "structured_toy")
    updated["data"]["importance_shard_dir"] = str(
        PROJECT_ROOT / "data" / "importance_shards" / "structured_toy_teacher"
    )
    updated["teacher_reference"]["checkpoint"] = str(
        PROJECT_ROOT
        / "runs"
        / "teacher_structured_toy_tiny_v1"
        / "checkpoints"
        / "teacher_world_model_step_000100.pt"
    )
    updated["learned_selector"]["checkpoint"] = str(
        PROJECT_ROOT
        / "runs"
        / "student_selector_structured_toy_v1"
        / "checkpoints"
        / "student_selector_step_000200.pt"
    )
    updated["output"]["run_root"] = str(PROJECT_ROOT / "runs")
    updated["output"]["run_name"] = "baseline_comparison_structured_toy_v1"
    updated["output"]["summary_json"] = str(run_dir / "baseline_summary.json")
    updated["output"]["summary_csv"] = str(run_dir / "baseline_summary.csv")
    updated["output"]["summary_md"] = str(run_dir / "baseline_summary.md")
    return updated


def _ensure_step8_inputs(config: dict[str, Any]) -> None:
    token_dir = Path(config["data"]["token_shard_dir"])
    importance_dir = Path(config["data"]["importance_shard_dir"])
    teacher_checkpoint = Path(config["teacher_reference"]["checkpoint"])
    selector_checkpoint = Path(config["learned_selector"]["checkpoint"])
    step7_checkpoint = (
        PROJECT_ROOT
        / "runs"
        / "student_world_model_structured_toy_v1"
        / "checkpoints"
        / "student_world_model_step_000300.pt"
    )
    token_ready = bool(list(token_dir.glob(config["data"].get("token_shard_glob", "tokens_shard_*.pt"))))
    importance_ready = bool(
        list(importance_dir.glob(config["data"].get("importance_shard_glob", "importance_shard_*.pt")))
    )
    if token_ready and importance_ready and teacher_checkpoint.exists() and selector_checkpoint.exists() and step7_checkpoint.exists():
        return
    run_step7_smoke()


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _policy_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in summary["rows"]:
        if row["policy"] not in rows:
            rows[row["policy"]] = row
    return rows


def _write_smoke_report(
    config: dict[str, Any],
    baseline_summary: dict[str, Any],
    pass_flag: bool,
) -> None:
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    baseline_md = Path(config["output"]["summary_md"]).read_text(encoding="utf-8")
    aggregate = baseline_summary["aggregate"]
    report = [
        "# Baseline Comparison Smoke Report",
        "",
        "Command: `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python scripts/smoke_test_baseline_comparison.py`",
        "",
        f"- run dir: `{run_dir}`",
        f"- policies: `{config['selection']['policies']}`",
        f"- summary json: `{config['output']['summary_json']}`",
        f"- summary csv: `{config['output']['summary_csv']}`",
        f"- summary md: `{config['output']['summary_md']}`",
        "",
        "## Baseline Summary Table",
        "",
        baseline_md,
        "",
        "## Learned Selector vs Random-K",
        "",
        "```json",
        json.dumps(aggregate["learned_vs_random"], indent=2),
        "```",
        "",
        "## Learned Selector vs Oracle-Key",
        "",
        "```json",
        json.dumps(aggregate["learned_vs_oracle"], indent=2),
        "```",
        "",
        "## Sanity Gate",
        "",
        "```json",
        json.dumps(aggregate["sanity_gate"], indent=2),
        "```",
        "",
        f"BASELINE_COMPARISON_SMOKE_PASS = {str(pass_flag).lower()}",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_step8_doc(
    config: dict[str, Any],
    baseline_summary: dict[str, Any],
    pass_flag: bool,
) -> None:
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    baseline_md = Path(config["output"]["summary_md"]).read_text(encoding="utf-8")
    aggregate = baseline_summary["aggregate"]
    doc = [
        "# STEP8 Baseline Comparison Report",
        "",
        "## 1. Goal",
        "",
        "Compare compressed Student world-model prediction on `structured_toy` under different token selection policies at the same token budget.",
        "",
        "## 2. Why Baselines Are Needed",
        "",
        "Step 7 showed the learned selector pipeline can train. Step 8 checks whether that selector is better than Random-K / Uniform-K and close to an Oracle-Key upper-bound under the same Student training protocol.",
        "",
        "## 3. Baselines",
        "",
        "- Full-token Teacher reference",
        "- Random-K Student",
        "- Uniform-K Student",
        "- Oracle-Key Student",
        "- Learned Selector Student",
        "",
        "## 4. Training Protocol",
        "",
        f"All Student baselines use `topK={config['selection']['topk']}`, the same `TokenCompressor`, the same `StudentWorldModel`, the same structured toy shards, and the same `max_steps={config['training']['max_steps']}` training budget. The teacher and learned selector are frozen references.",
        "",
        "## 5. Metrics",
        "",
        "- future_mse",
        "- teacher_mse",
        "- student_teacher_ratio",
        "- token_retention_ratio",
        "- selected_top1_hit_rate",
        "- selected_topk_hit_rate",
        "- selected_key_coverage",
        "",
        "## 6. Results Table",
        "",
        baseline_md,
        "",
        "## 7. Sanity Gate",
        "",
        "```json",
        json.dumps(aggregate["sanity_gate"], indent=2),
        "```",
        "",
        f"Smoke gate pass: `{str(pass_flag).lower()}`",
        "",
        "## 8. Pytest Result",
        "",
        "Run `/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab python -m pytest tests -q` after Step 8 changes. The final local summary records the observed result.",
        "",
        "## 9. Git Commit",
        "",
        "Branch: `feature/tgpawb-step3-token-extraction`. Commit hash and push status are recorded after commit/push. No PR is created in this stage.",
        "",
        "## 10. What Was Not Done",
        "",
        "- no real dataset download",
        "- no real V-JEPA / VideoMAE encoder",
        "- no large model download",
        "- no VLM grounding",
        "- no RL / policy optimization",
        "- no action-conditioned world model",
        "- no generated `.pt` committed",
        "- no checkpoint committed",
        "- no password or token saved",
        "- no PR created",
        "",
        "## 11. Next Step Recommendation",
        "",
        "Step 9 should prepare a minimal real-video subset and loader before introducing frozen VideoMAE/V-JEPA token extraction.",
        "",
        "## Artifact Paths",
        "",
        f"- run dir: `{run_dir}`",
        f"- baseline summary json: `{config['output']['summary_json']}`",
        f"- baseline summary csv: `{config['output']['summary_csv']}`",
        f"- baseline summary md: `{config['output']['summary_md']}`",
    ]
    STEP8_DOC_PATH.write_text("\n".join(doc) + "\n", encoding="utf-8")


def main() -> None:
    config = _project_config(load_yaml(PROJECT_ROOT / "configs" / "baseline_comparison_structured_toy.yaml"))
    _ensure_step8_inputs(config)
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    if run_dir.exists():
        shutil.rmtree(run_dir)

    train_result = train_baseline_comparison(config)
    eval_result = evaluate_baseline_comparison(run_dir)
    rows_by_policy = _policy_rows(eval_result)
    required_policies = {"random_k", "uniform_k", "oracle_key", "learned_selector"}
    learned = rows_by_policy.get("learned_selector")
    checks = {
        "summary_json_exists": Path(config["output"]["summary_json"]).exists(),
        "summary_csv_exists": Path(config["output"]["summary_csv"]).exists(),
        "summary_md_exists": Path(config["output"]["summary_md"]).exists(),
        "contains_required_policies": required_policies.issubset(rows_by_policy),
        "learned_selector_present": learned is not None,
        "learned_selected_topk_hit_rate": bool(learned and learned["selected_topk_hit_rate"] >= 0.8),
        "learned_selected_key_coverage": bool(learned and learned["selected_key_coverage"] >= 0.8),
        "learned_student_teacher_ratio_finite": bool(learned and _finite(learned["student_teacher_ratio"])),
        "sanity_gate": bool(eval_result["aggregate"]["sanity_gate"]["pass"]),
    }
    pass_flag = all(checks.values())
    eval_result["smoke_checks"] = checks
    Path(config["output"]["summary_json"]).write_text(json.dumps(eval_result, indent=2), encoding="utf-8")
    _write_smoke_report(config, eval_result, pass_flag)
    _write_step8_doc(config, eval_result, pass_flag)
    print(REPORT_PATH.read_text(encoding="utf-8"))
    if not pass_flag:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
