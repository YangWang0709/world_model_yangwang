"""Validation wrapper for Step 16 BAIR downstream utilization ablation."""

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

from eval.eval_bair_downstream_utilization_ablation import (
    load_yaml,
    render_downstream_markdown,
    write_downstream_utilization_summaries,
)
from scripts.run_bair_500_64_scale_validation import ensure_resource_limits, resource_snapshot
from training.run_bair_downstream_utilization_ablation import (
    DEFAULT_CONFIG,
    is_step16_owned_output_path,
    run_downstream_utilization_ablation,
    validate_required_inputs,
)


REPORT_PATH = PROJECT_ROOT / "docs" / "BAIR_DOWNSTREAM_UTILIZATION_ABLATION_REPORT.md"
STEP_DOC_PATH = PROJECT_ROOT / "docs" / "STEP16_BAIR_DOWNSTREAM_UTILIZATION_ABLATION.md"


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
        "output_tail": "\n".join(output.strip().splitlines()[-30:]),
    }


def _env_guard() -> dict[str, bool]:
    import importlib.util

    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def _required_outputs(run_dir: Path) -> list[Path]:
    return [
        run_dir / "phase_a_summary.json",
        run_dir / "phase_a_summary.csv",
        run_dir / "phase_a_summary.md",
        run_dir / "phase_b_summary.json",
        run_dir / "phase_b_summary.csv",
        run_dir / "phase_b_summary.md",
        run_dir / "downstream_utilization_summary.json",
        run_dir / "downstream_utilization_summary.md",
    ]


def _assert_required_outputs(run_dir: Path) -> None:
    missing = [str(path) for path in _required_outputs(run_dir) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Step16 outputs: {missing}")


def _safe_reset_step16_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        if not is_step16_owned_output_path(run_dir):
            raise ValueError(f"Refusing to delete non-Step16 run dir: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.8f}"
    except (TypeError, ValueError):
        return str(value)


def write_validation_report(summary: dict[str, Any], pytest_result: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# BAIR Downstream Utilization Ablation Report",
        "",
        "Command: `python scripts/smoke_test_bair_downstream_utilization_ablation.py`",
        "",
        "## Run Dirs",
        "",
        f"- Step16 run: `{summary.get('run_dir')}`",
        f"- Phase A: `{Path(summary.get('run_dir', '')) / 'phase_a'}`",
        f"- Phase B: `{Path(summary.get('run_dir', '')) / 'phase_b'}`",
        "",
        "## Fixed Step15 Inputs",
        "",
    ]
    for key, value in summary.get("input_paths", {}).items():
        report.append(f"- {key}: `{value}`")
    report.extend(
        [
            "",
            "## Unified Summary",
            "",
            render_downstream_markdown(summary),
            "",
            "## Interpretation",
            "",
            f"- best variant: `{summary.get('phase_b_best_mse_variant')}`",
            f"- best Step16 MSE: `{_fmt(summary.get('phase_b_best_mse_mean'))}`",
            f"- current compressor bottleneck signal: `{str(bool(summary.get('best_beats_step15_learned'))).lower()}` for beating Step15 learned; `{str(bool(summary.get('best_beats_step15_uniform'))).lower()}` for beating uniform",
            f"- hybrid learned+uniform reached Phase B: `{str(summary.get('hybrid_learned_uniform_reached_phase_b')).lower()}`",
            f"- cross-attention reached Phase B: `{str(summary.get('cross_attention_reached_phase_b')).lower()}`",
            "- Step17 recommendation: use the best Step16 utilizer if it beats uniform; otherwise prioritize action-conditioned world model.",
            "",
            "## Pytest",
            "",
            "```text",
            pytest_result.get("output_tail", ""),
            "```",
            "",
            f"BAIR_DOWNSTREAM_UTILIZATION_ABLATION_PASS = {str(summary.get('sanity_gate_pass')).lower()}",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")


def write_step_doc(summary: dict[str, Any], pytest_result: dict[str, Any]) -> None:
    lines = [
        "# STEP16 BAIR Downstream Utilization / Compressor Ablation",
        "",
        "## 1. Goal",
        "",
        "Diagnose the Step15 gap where the learned selector selects higher Teacher-importance tokens but the downstream StudentWorldModel does not reliably beat uniform selection.",
        "",
        "## 2. Motivation",
        "",
        "- learned selected importance > random/uniform",
        "- learned MSE is only slightly better than random",
        "- learned MSE is worse than uniform",
        "- the likely bottleneck is selected-token utilization in the compressor / StudentWorldModel.",
        "",
        "## 3. Cloud Server Decision",
        "",
        "No cloud server is required for this stage because it reuses Step15 BAIR 1000/128 tokens, importance shards, Teacher checkpoint, and selector checkpoints. A larger 4090 / 48GB server is only recommended if CUDA OOM persists after reducing batch size.",
        "",
        "## 4. Input",
        "",
    ]
    for key, value in summary.get("input_paths", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## 5. Variants",
            "",
            "- current_perceiver_like",
            "- mean_pool_selected",
            "- attention_pool_selected",
            "- selector_score_weighted_pool",
            "- transformer_encoder_selected",
            "- cross_attention_latent_bottleneck",
            "- hybrid_learned8_uniform8_perceiver",
            "- hybrid_learned8_uniform8_cross_attention",
            "",
            "## 6. Phase A",
            "",
            Path(summary.get("run_dir", "")) .joinpath("phase_a_summary.md").read_text(encoding="utf-8")
            if summary.get("run_dir") and Path(summary["run_dir"]).joinpath("phase_a_summary.md").exists()
            else "Phase A summary is unavailable.",
            "",
            "## 7. Phase B",
            "",
            Path(summary.get("run_dir", "")) .joinpath("phase_b_summary.md").read_text(encoding="utf-8")
            if summary.get("run_dir") and Path(summary["run_dir"]).joinpath("phase_b_summary.md").exists()
            else "Phase B summary is unavailable.",
            "",
            "## 8. Comparison vs Step15",
            "",
            f"- best Step16 vs Step15 learned delta: `{_fmt((summary.get('phase_b_best_mse_mean') or 0) - summary.get('step15_learned_mse_mean')) if summary.get('phase_b_best_mse_mean') is not None else 'n/a'}`",
            f"- best Step16 vs Step15 random delta: `{_fmt((summary.get('phase_b_best_mse_mean') or 0) - summary.get('step15_random_mse_mean')) if summary.get('phase_b_best_mse_mean') is not None else 'n/a'}`",
            f"- best Step16 vs Step15 uniform delta: `{_fmt((summary.get('phase_b_best_mse_mean') or 0) - summary.get('step15_uniform_mse')) if summary.get('phase_b_best_mse_mean') is not None else 'n/a'}`",
            f"- best Step16 vs teacher_importance_topk delta: `{_fmt(summary.get('gap_to_teacher_importance_topk'))}`",
            f"- hybrid reached Phase B: `{str(summary.get('hybrid_learned_uniform_reached_phase_b')).lower()}`",
            f"- cross-attention reached Phase B: `{str(summary.get('cross_attention_reached_phase_b')).lower()}`",
            "",
            "## 9. Interpretation",
            "",
            "If hybrid learned+uniform is best, the learned tokens likely need additional global coverage. If cross-attention is best, the current compressor is likely too weak for non-uniform selected tokens. If no variant beats uniform, downstream architecture alone is not enough and Step17 should prioritize action-conditioned dynamics.",
            "",
            "## 10. Sanity Gate",
            "",
            "```json",
            json.dumps(summary.get("sanity_gate", {}), indent=2),
            "```",
            "",
            "## 11. Resource Usage",
            "",
            "```json",
            json.dumps(summary.get("resource_summary", {}), indent=2),
            "```",
            "",
            "## 12. Pytest",
            "",
            f"- command: `{pytest_result.get('command')}`",
            f"- passed: `{str(pytest_result.get('passed')).lower()}`",
            "",
            "## 13. Git Commit",
            "",
            "Commit hash is recorded in the final local summary after push.",
            "",
            "## 14. What Was Not Done",
            "",
            "- no new model download",
            "- no new dataset download",
            "- no BAIR re-download",
            "- no VideoMAE extraction",
            "- no VideoMAE training",
            "- no Teacher retraining",
            "- no importance regeneration",
            "- no selector retraining",
            "- no Step11-15 overwrite",
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
            "## 15. Next Step Recommendation",
            "",
            "Use the best Step16 utilizer if it beats Step15 uniform. If it does not, proceed to an action-conditioned StudentWorldModel in Step17.",
            "",
        ]
    )
    STEP_DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_yaml(DEFAULT_CONFIG)
    run_dir = Path(config["output"]["run_root"]) / config["output"]["run_name"]
    validate_required_inputs(config)
    before = resource_snapshot()
    ensure_resource_limits(config, before)
    _safe_reset_step16_run_dir(run_dir)
    run_downstream_utilization_ablation(DEFAULT_CONFIG)
    _assert_required_outputs(run_dir)
    pytest_result = _run_pytest()
    env_guard = _env_guard()
    previous = json.loads((run_dir / "downstream_utilization_summary.json").read_text(encoding="utf-8"))
    summary = write_downstream_utilization_summaries(
        run_dir=run_dir,
        config=config,
        step15_aggregate_path=config["step15_reference"]["baseline_aggregate"],
        resource_summary=previous.get("resource_summary", {}),
        pytest_result=pytest_result,
        env_guard=env_guard,
    )
    write_validation_report(summary, pytest_result)
    write_step_doc(summary, pytest_result)
    print(REPORT_PATH.read_text(encoding="utf-8"))
    print(f"BAIR_DOWNSTREAM_UTILIZATION_ABLATION_PASS = {str(summary.get('sanity_gate_pass')).lower()}")
    if not summary.get("sanity_gate_pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
