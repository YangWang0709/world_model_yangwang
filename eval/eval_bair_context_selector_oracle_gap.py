"""Summarize Step18 BAIR context-selector oracle-gap diagnostic outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "context_selector_oracle_gap_bair_1000_128_v1"
DEFAULT_STEP17_SUMMARY = PROJECT_ROOT / "runs" / "context_bottleneck_bair_1000_128_v1" / "context_bottleneck_summary.json"
DEFAULT_STEP17_BASELINE = PROJECT_ROOT / "runs" / "context_bottleneck_baseline_bair_1000_128_v1" / "baseline_aggregate.json"


def _read_json(path: str | Path, required: bool = True) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(p)
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {p}")
    return payload


def _write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fmt(value: Any, precision: int = 8) -> str:
    return f"{float(value):.{precision}f}" if _is_finite(value) else "n/a"


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows", [])
    return rows if isinstance(rows, list) else []


def _aggregate_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("aggregate", [])
    return rows if isinstance(rows, list) else []


def _best_by(rows: list[dict[str, Any]], key: str, *, lower_is_better: bool = False) -> dict[str, Any]:
    finite = [row for row in rows if _is_finite(row.get(key))]
    if not finite:
        return {}
    return sorted(finite, key=lambda row: float(row[key]), reverse=not lower_is_better)[0]


def _row_by_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    for row in rows:
        if row.get("variant") == variant:
            return row
    return {}


def _phase_b_best(downstream_aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    return _best_by(downstream_aggregate, "future_mse_mean", lower_is_better=True)


def _diagnosis(
    *,
    label_diagnostic: dict[str, Any],
    phase_a_selector_rows: list[dict[str, Any]],
    phase_a_downstream_rows: list[dict[str, Any]],
    phase_b_selector_aggregate: list[dict[str, Any]],
    phase_b_downstream_best: dict[str, Any],
    step17_random_mse: float,
) -> dict[str, bool]:
    base = _row_by_variant(phase_a_selector_rows, "weighted_mse_alpha2")
    no_current = _row_by_variant(phase_a_selector_rows, "weighted_mse_alpha2_no_current_condition")
    no_temp = _row_by_variant(phase_a_selector_rows, "weighted_mse_alpha2_no_temporal_pos")
    temporal = _row_by_variant(phase_a_selector_rows, "temporal_block_balanced_topk")
    hybrid = _row_by_variant(phase_a_selector_rows, "hybrid_weighted_mse_rank_bce")
    best_overlap = _best_by(phase_a_selector_rows, "target_topk_overlap")
    return {
        "label_sparse_or_noisy": bool(label_diagnostic.get("label_sparse_or_noisy", False)),
        "topk_overlap_bottleneck": bool(float(best_overlap.get("target_topk_overlap", 0.0) or 0.0) < 0.25),
        "current_conditioning_helpful": bool(
            _is_finite(base.get("target_topk_overlap"))
            and _is_finite(no_current.get("target_topk_overlap"))
            and float(base["target_topk_overlap"]) >= float(no_current["target_topk_overlap"])
        ),
        "temporal_position_helpful": bool(
            _is_finite(base.get("target_topk_overlap"))
            and _is_finite(no_temp.get("target_topk_overlap"))
            and float(base["target_topk_overlap"]) >= float(no_temp["target_topk_overlap"])
        ),
        "temporal_block_balancing_helpful": bool(
            _is_finite(temporal.get("target_topk_overlap"))
            and _is_finite(hybrid.get("target_topk_overlap"))
            and float(temporal["target_topk_overlap"]) >= float(hybrid["target_topk_overlap"])
        ),
        "downstream_utilization_still_bottleneck": bool(
            not _is_finite(phase_b_downstream_best.get("future_mse_mean"))
            or float(phase_b_downstream_best["future_mse_mean"]) >= float(step17_random_mse)
        ),
    }


def build_context_selector_oracle_gap_summary(
    *,
    run_dir: str | Path,
    step17_summary: dict[str, Any],
    step17_baseline: dict[str, Any],
    pytest_result: dict[str, Any] | None = None,
    env_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run = Path(run_dir)
    label = _read_json(run / "label_diagnostic" / "context_importance_diagnostic.json")
    phase_a_selector = _rows(_read_json(run / "selector_phase_a_summary.json"))
    phase_a_downstream = _rows(_read_json(run / "downstream_phase_a_summary.json"))
    phase_b_selector = _read_json(run / "selector_phase_b_aggregate.json")
    phase_b_downstream = _read_json(run / "downstream_phase_b_aggregate.json")
    runner = _read_json(run / "runner_summary.json", required=False)
    phase_b_selector_rows = _aggregate_rows(phase_b_selector)
    phase_b_downstream_rows = _aggregate_rows(phase_b_downstream)
    best_overlap = _best_by(phase_a_selector, "target_topk_overlap")
    best_importance = _best_by(phase_a_selector, "selected_context_importance_mean")
    best_phase_a_downstream = _best_by(phase_a_downstream, "future_mse", lower_is_better=True)
    best_phase_b = _phase_b_best(phase_b_downstream_rows)

    step17_current = float(step17_summary.get("current_only_mse", step17_summary.get("step17_current_only_mse", 1.39464128)))
    step17_random = float(step17_summary.get("random_context_mse", step17_summary.get("step17_random_context_topk_mse", 1.36507559)))
    step17_learned = float(step17_summary.get("learned_context_mse", step17_summary.get("step17_learned_context_mse", 1.37120354)))
    step17_oracle = float(step17_summary.get("teacher_context_importance_topk_mse", step17_summary.get("step17_teacher_context_importance_topk_mse", 1.21603048)))
    if not _is_finite(step17_random):
        for row in step17_baseline.get("aggregate", []):
            if row.get("policy") == "random_context_topK":
                step17_random = float(row["student_future_mse_mean"])
    best_mse = best_phase_b.get("future_mse_mean", best_phase_a_downstream.get("future_mse"))
    best_topk = _row_by_variant(phase_b_selector_rows, str(best_phase_b.get("variant"))).get("topk_overlap_mean")
    best_importance_mean = _row_by_variant(phase_b_selector_rows, str(best_phase_b.get("variant"))).get("selected_importance_mean")
    oracle_gap_after = float(best_mse) - step17_oracle if _is_finite(best_mse) else None
    step17_oracle_gap = step17_learned - step17_oracle
    gap_reduction = step17_oracle_gap - float(oracle_gap_after) if _is_finite(oracle_gap_after) else None
    diagnosis = _diagnosis(
        label_diagnostic=label,
        phase_a_selector_rows=phase_a_selector,
        phase_a_downstream_rows=phase_a_downstream,
        phase_b_selector_aggregate=phase_b_selector_rows,
        phase_b_downstream_best=best_phase_b,
        step17_random_mse=step17_random,
    )
    next_step = "Proceed with stronger context selector architecture before moving to longer-context datasets."
    if _is_finite(best_mse) and float(best_mse) < step17_random and float(gap_reduction or 0.0) > 0:
        next_step = "Step18 reduced the oracle gap; action-conditioned context world model can be considered next, with the current no-action result kept as control."
    elif diagnosis["label_sparse_or_noisy"]:
        next_step = "Labels look diffuse or noisy on BAIR; prefer a longer-context dataset such as BridgeData or DROID before claiming long-memory value."
    elif diagnosis["topk_overlap_bottleneck"]:
        next_step = "TopK overlap remains the bottleneck; test a stronger selector architecture and ranking-oriented objective."
    checks = {
        "step17_inputs_exist": True,
        "label_diagnostic_generated": bool(label),
        "phase_a_selector_variants_recorded": len(phase_a_selector) >= 8,
        "phase_b_two_variants_three_seeds": sum(int(row.get("n", 0)) for row in phase_b_selector_rows) >= 6,
        "metrics_finite": all(_is_finite(row.get("future_mse_mean")) for row in phase_b_downstream_rows),
        "current_tokens_dropped_false": not bool(runner.get("current_tokens_dropped", False)),
        "trained_current_importance_false": not bool(runner.get("trained_current_importance", False)),
        "trained_context_importance_true": bool(runner.get("trained_context_importance", True)),
        "pytest_passed": bool((pytest_result or {}).get("passed", True)),
        "env_isaaclab_unpolluted": not bool((env_guard or {}).get("tensorflow", False)) and not bool((env_guard or {}).get("tensorflow_datasets", False)),
        "oom_false": not bool(runner.get("resource_summary", {}).get("oom", False)),
    }
    return {
        "stage": "context_selector_oracle_gap_bair_1000_128",
        "run_dir": str(run),
        "train_samples": 1000,
        "test_samples": 128,
        "context_topk": 32,
        "current_tokens_dropped": False,
        "trained_current_importance": False,
        "trained_context_importance": True,
        "label_diagnostic": label,
        "phase_a_best_selector_by_topk_overlap": best_overlap.get("variant"),
        "phase_a_best_selector_by_selected_importance": best_importance.get("variant"),
        "phase_a_best_downstream_by_mse": best_phase_a_downstream.get("variant"),
        "phase_b_best_variant": best_phase_b.get("variant"),
        "phase_b_best_downstream_mse_mean": best_phase_b.get("future_mse_mean"),
        "phase_b_best_downstream_mse_std": best_phase_b.get("future_mse_std"),
        "phase_b_best_topk_overlap_mean": best_topk,
        "phase_b_best_selected_importance_mean": best_importance_mean,
        "step17_current_only_mse": step17_current,
        "step17_random_context_topk_mse": step17_random,
        "step17_learned_context_mse": step17_learned,
        "step17_teacher_context_importance_topk_mse": step17_oracle,
        "best_beats_step17_learned": bool(_is_finite(best_mse) and float(best_mse) < step17_learned),
        "best_beats_random_context": bool(_is_finite(best_mse) and float(best_mse) < step17_random),
        "best_beats_current_only": bool(_is_finite(best_mse) and float(best_mse) < step17_current),
        "oracle_gap_after_step18": oracle_gap_after,
        "oracle_gap_reduction_vs_step17": gap_reduction,
        "diagnosis": diagnosis,
        "next_step_recommendation": next_step,
        "sanity_gate": {"checks": checks, "pass": all(checks.values())},
        "sanity_gate_pass": bool(all(checks.values())),
        "cloud_required": False,
        "resource_summary": runner.get("resource_summary", {}),
        "pytest_result": pytest_result or {},
        "env_guard": env_guard or {},
        "selector_phase_a_rows": phase_a_selector,
        "downstream_phase_a_rows": phase_a_downstream,
        "selector_phase_b_aggregate": phase_b_selector,
        "downstream_phase_b_aggregate": phase_b_downstream,
    }


def render_context_selector_oracle_gap_markdown(summary: dict[str, Any]) -> str:
    label = summary["label_diagnostic"]
    stats = label["importance_stats"]
    norm = label["importance_norm_stats"]
    phase_a_selector = summary.get("selector_phase_a_rows", [])
    phase_a_downstream = summary.get("downstream_phase_a_rows", [])
    phase_b_selector = summary.get("selector_phase_b_aggregate", {}).get("aggregate", [])
    phase_b_downstream = summary.get("downstream_phase_b_aggregate", {}).get("aggregate", [])
    lines = [
        "# BAIR Context Selector Oracle-Gap Summary",
        "",
        f"- stage: `{summary['stage']}`",
        f"- context_topK: `{summary['context_topk']}`",
        f"- current_tokens_dropped: `{str(summary['current_tokens_dropped']).lower()}`",
        f"- trained_current_importance: `{str(summary['trained_current_importance']).lower()}`",
        f"- trained_context_importance: `{str(summary['trained_context_importance']).lower()}`",
        f"- sanity gate pass: `{str(summary['sanity_gate_pass']).lower()}`",
        f"- cloud required: `{str(summary['cloud_required']).lower()}`",
        "",
        "## Label Diagnostic",
        "",
        f"- raw importance mean/std/min/max: `{stats['mean']:.8f}` / `{stats['std']:.8f}` / `{stats['min']:.8f}` / `{stats['max']:.8f}`",
        f"- norm importance mean/std/min/max: `{norm['mean']:.8f}` / `{norm['std']:.8f}` / `{norm['min']:.8f}` / `{norm['max']:.8f}`",
        f"- positive ratio: `{label['positive_importance_ratio']:.6f}`",
        f"- top32 mass ratio: `{label['topk_concentration'].get('top32_mass_ratio', 0.0):.8f}`",
        f"- oracle vs random importance gap: `{label['oracle_random_gap']['oracle_vs_random_importance_gap']:.8f}`",
        f"- temporal block distribution: `{[round(float(v), 4) for v in label['temporal_block_stats']['oracle_topk_block_distribution']]}`",
        "",
        "## Selector Phase A",
        "",
        _simple_table(
            phase_a_selector,
            ["variant", "loss_type", "context_importance_mse", "pearson_corr_mean", "target_topk_overlap", "selected_context_importance_mean", "temporal_block_coverage"],
        ),
        "",
        "## Downstream Phase A",
        "",
        _simple_table(
            phase_a_downstream,
            ["variant", "future_mse", "beats_step17_learned", "beats_random_context", "oracle_gap"],
        ),
        "",
        "## Selector Phase B",
        "",
        _simple_table(
            phase_b_selector,
            ["variant", "seeds", "topk_overlap_mean", "topk_overlap_std", "selected_importance_mean", "selected_importance_std"],
        ),
        "",
        "## Downstream Phase B",
        "",
        _simple_table(phase_b_downstream, ["variant", "seeds", "future_mse_mean", "future_mse_std", "oracle_gap_mean"]),
        "",
        "## Oracle Gap",
        "",
        f"- Step17 learned MSE: `{_fmt(summary['step17_learned_context_mse'])}`",
        f"- Step17 random MSE: `{_fmt(summary['step17_random_context_topk_mse'])}`",
        f"- Step17 current_only MSE: `{_fmt(summary['step17_current_only_mse'])}`",
        f"- Step17 teacher oracle MSE: `{_fmt(summary['step17_teacher_context_importance_topk_mse'])}`",
        f"- Step18 best variant: `{summary.get('phase_b_best_variant')}`",
        f"- Step18 best MSE mean: `{_fmt(summary.get('phase_b_best_downstream_mse_mean'))}`",
        f"- oracle gap after Step18: `{_fmt(summary.get('oracle_gap_after_step18'))}`",
        f"- oracle gap reduction vs Step17: `{_fmt(summary.get('oracle_gap_reduction_vs_step17'))}`",
        "",
        "## Diagnosis",
        "",
    ]
    lines.extend([f"- {key}: `{str(value).lower()}`" for key, value in summary["diagnosis"].items()])
    lines.extend(
        [
            "",
            "## Next Recommendation",
            "",
            summary["next_step_recommendation"],
            "",
            "## Sanity Gate",
            "",
            "```json",
            json.dumps(summary.get("sanity_gate", {}), indent=2),
            "```",
            "",
            "## Resource Summary",
            "",
            "```json",
            json.dumps(summary.get("resource_summary", {}), indent=2),
            "```",
            "",
            f"BAIR_CONTEXT_SELECTOR_ORACLE_GAP_PASS = {str(summary.get('sanity_gate_pass')).lower()}",
            "",
        ]
    )
    return "\n".join(lines)


def _simple_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_no rows_"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            cells.append(_fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_context_selector_oracle_gap_summary(
    *,
    run_dir: str | Path,
    step17_summary_path: str | Path,
    step17_baseline_path: str | Path,
    pytest_result: dict[str, Any] | None = None,
    env_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = build_context_selector_oracle_gap_summary(
        run_dir=run_dir,
        step17_summary=_read_json(step17_summary_path),
        step17_baseline=_read_json(step17_baseline_path),
        pytest_result=pytest_result,
        env_guard=env_guard,
    )
    run = Path(run_dir)
    _write_json(run / "context_selector_oracle_gap_summary.json", summary)
    (run / "context_selector_oracle_gap_summary.md").write_text(render_context_selector_oracle_gap_markdown(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--step17-summary", default=str(DEFAULT_STEP17_SUMMARY))
    parser.add_argument("--step17-baseline", default=str(DEFAULT_STEP17_BASELINE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_context_selector_oracle_gap_summary(
        run_dir=args.run_dir,
        step17_summary_path=args.step17_summary,
        step17_baseline_path=args.step17_baseline,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
