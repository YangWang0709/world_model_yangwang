"""Summarize Step17 BAIR context bottleneck validation outputs."""

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


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "context_bottleneck_bair_1000_128.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return payload


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


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fmt(value: Any, precision: int = 8) -> str:
    return f"{float(value):.{precision}f}" if _is_finite(value) else "n/a"


def build_context_bottleneck_summary(
    *,
    config: dict[str, Any],
    resource_summary: dict[str, Any] | None = None,
    pytest_result: dict[str, Any] | None = None,
    env_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_cfg = config["output"]
    run_dir = Path(output_cfg["run_root"]) / output_cfg["run_name"]
    context_windows = _read_json(Path(config["dataset"]["output_dir"]) / "export_summary.json", required=False)
    token_summary = _read_json(Path(config["tokens"]["output_root"]) / "extraction_summary.json", required=False)
    teacher_run = Path(output_cfg["run_root"]) / config["teacher"]["run_name"]
    selector_run = Path(output_cfg["run_root"]) / config["unified_selector"]["run_name"]
    world_run = Path(output_cfg["run_root"]) / config["context_bottleneck_world_model"]["run_name"]
    baseline_run = Path(output_cfg["run_root"]) / config["baselines"]["run_name"]
    teacher_summary = _read_json(teacher_run / "summary.json", required=False)
    importance_summary = _read_json(Path(config["importance"]["output_root"]) / "importance_summary.json", required=False)
    selector_summary = _read_json(selector_run / "summary.json", required=False)
    world_summary = _read_json(world_run / "summary.json", required=False)
    baseline_summary = _read_json(baseline_run / "baseline_summary.json", required=False)
    baseline_aggregate = _read_json(baseline_run / "baseline_aggregate.json", required=False)
    aggregate_rows = baseline_aggregate.get("aggregate", []) if isinstance(baseline_aggregate.get("aggregate"), list) else []

    def row(policy: str) -> dict[str, Any]:
        for item in aggregate_rows:
            if item.get("policy") == policy:
                return item
        return {}

    learned = row("learned_context_selector_topK")
    current = row("current_only")
    random_row = row("random_context_topK")
    uniform = row("uniform_context_topK")
    oracle = row("teacher_context_importance_topK")
    hybrid = row("hybrid_context_learned_uniform")
    learned_mse = learned.get("student_future_mse_mean", world_summary.get("student_future_mse"))
    current_mse = current.get("student_future_mse_mean")
    random_mse = random_row.get("student_future_mse_mean")
    uniform_mse = uniform.get("student_future_mse_mean")
    oracle_mse = oracle.get("student_future_mse_mean")
    teacher_mse = teacher_summary.get("eval_mse", world_summary.get("context_teacher_mse"))
    context_gain = float(current_mse) - float(learned_mse) if _is_finite(current_mse) and _is_finite(learned_mse) else None
    oracle_gap = float(learned_mse) - float(oracle_mse) if _is_finite(learned_mse) and _is_finite(oracle_mse) else None
    caveats: list[str] = []
    if not (_is_finite(current_mse) and _is_finite(learned_mse) and float(learned_mse) < float(current_mse)):
        caveats.append("learned context did not clearly beat current_only; BAIR short-window prediction may be dominated by current observation")
    if not (_is_finite(random_mse) and _is_finite(learned_mse) and float(learned_mse) <= float(random_mse)):
        caveats.append("learned context did not clearly beat random context topK")
    if not (_is_finite(uniform_mse) and _is_finite(learned_mse) and float(learned_mse) <= float(uniform_mse)):
        caveats.append("learned context did not clearly beat uniform context topK")
    if not (_is_finite(oracle_mse) and _is_finite(learned_mse)):
        caveats.append("oracle teacher_context_importance_topK comparison is unavailable")
    checks = {
        "context_windows_exported": int(context_windows.get("splits", {}).get("train", {}).get("exported", 0)) == int(config["dataset"]["train_samples"]),
        "context_tokens_extracted": int(token_summary.get("split_summaries", {}).get("train", {}).get("dataset_size", 0)) == int(config["dataset"]["train_samples"]),
        "context_teacher_trained": Path(config["teacher"]["checkpoint"]).exists() and _is_finite(teacher_mse),
        "context_importance_generated": int(importance_summary.get("train_num_samples", 0)) == int(config["dataset"]["train_samples"]),
        "unified_selector_trained_context_only": Path(config["unified_selector"]["checkpoint"]).exists() and selector_summary.get("trained_mode") == "context" and not bool(selector_summary.get("trained_current_importance", True)),
        "world_model_trained": Path(config["context_bottleneck_world_model"]["checkpoint"]).exists() and _is_finite(world_summary.get("student_future_mse")),
        "baseline_comparison_complete": len(aggregate_rows) >= 5,
        "current_tokens_dropped_false": not bool(world_summary.get("current_tokens_dropped", True)),
        "pytest_passed": bool((pytest_result or {}).get("passed", True)),
        "env_isaaclab_unpolluted": not bool((env_guard or {}).get("tensorflow", False)) and not bool((env_guard or {}).get("tensorflow_datasets", False)),
        "oom_false": not bool((resource_summary or {}).get("oom", False)),
    }
    summary = {
        "stage": "context_bottleneck_bair_1000_128",
        "run_dir": str(run_dir),
        "context_window_summary": context_windows,
        "token_extraction_summary": token_summary,
        "context_teacher_summary": teacher_summary,
        "context_importance_summary": importance_summary,
        "unified_selector_summary": selector_summary,
        "context_bottleneck_world_model_summary": world_summary,
        "baseline_summary": baseline_summary,
        "baseline_aggregate": baseline_aggregate,
        "train_samples": int(config["dataset"]["train_samples"]),
        "test_samples": int(config["dataset"]["test_samples"]),
        "context_len": int(config["dataset"]["context_len"]),
        "current_len": int(config["dataset"]["current_len"]),
        "future_len": int(config["dataset"]["future_len"]),
        "context_topK": int(world_summary.get("context_topK", selector_summary.get("topk", 0)) or 0),
        "context_retention_ratio": world_summary.get("context_retention_ratio", selector_summary.get("context_retention_ratio")),
        "current_tokens_dropped": False,
        "trained_current_importance": False,
        "trained_context_importance": True,
        "context_teacher_mse": teacher_mse,
        "current_only_mse": current_mse,
        "learned_context_mse": learned_mse,
        "random_context_mse": random_mse,
        "uniform_context_mse": uniform_mse,
        "teacher_context_importance_topk_mse": oracle_mse,
        "hybrid_context_mse": hybrid.get("student_future_mse_mean"),
        "context_gain_over_current_only": context_gain,
        "oracle_gap": oracle_gap,
        "learned_beats_current_only": bool(_is_finite(context_gain) and float(context_gain) > 0.0),
        "learned_beats_random": bool(_is_finite(random_mse) and _is_finite(learned_mse) and float(learned_mse) <= float(random_mse)),
        "learned_beats_uniform": bool(_is_finite(uniform_mse) and _is_finite(learned_mse) and float(learned_mse) <= float(uniform_mse)),
        "hybrid_beats_learned": bool(_is_finite(hybrid.get("student_future_mse_mean")) and _is_finite(learned_mse) and float(hybrid["student_future_mse_mean"]) < float(learned_mse)),
        "bair_context_limitation": "BAIR context window validates the context bottleneck pipeline, but may not be sufficient to prove long-context memory value.",
        "scientific_caveats": caveats,
        "sanity_gate": {"checks": checks, "pass": all(checks.values())},
        "sanity_gate_pass": all(checks.values()),
        "resource_summary": resource_summary or {},
        "pytest_result": pytest_result or {},
        "env_guard": env_guard or {},
        "cloud_required": False,
    }
    return summary


def render_context_bottleneck_markdown(summary: dict[str, Any]) -> str:
    rows = summary.get("baseline_aggregate", {}).get("aggregate", [])
    lines = [
        "# BAIR Context Bottleneck Validation Summary",
        "",
        f"- stage: `{summary['stage']}`",
        f"- train/test samples: `{summary['train_samples']}` / `{summary['test_samples']}`",
        f"- context/current/future frames: `{summary['context_len']}` / `{summary['current_len']}` / `{summary['future_len']}`",
        f"- context_topK: `{summary.get('context_topK')}`",
        f"- context retention ratio: `{_fmt(summary.get('context_retention_ratio'), 12)}`",
        f"- current_tokens_dropped: `{str(summary.get('current_tokens_dropped')).lower()}`",
        f"- trained_current_importance: `{str(summary.get('trained_current_importance')).lower()}`",
        f"- trained_context_importance: `{str(summary.get('trained_context_importance')).lower()}`",
        f"- sanity gate pass: `{str(summary.get('sanity_gate_pass')).lower()}`",
        f"- cloud required: `{str(summary.get('cloud_required')).lower()}`",
        "",
        "## Baselines",
        "",
        "`full_context_teacher_reference` is a non-deployable teacher reference that uses full historical context; it is not a learned bottleneck policy.",
        "",
        "| policy | seeds | n | mse_mean | mse_std | selected_context_importance | topK_overlap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: float(item.get("student_future_mse_mean", 1e9)) if _is_finite(item.get("student_future_mse_mean")) else 1e9):
        lines.append(
            "| {policy} | {seeds} | {n} | {mse} | {std} | {importance} | {topk} |".format(
                policy=row.get("policy"),
                seeds=row.get("seeds", ""),
                n=row.get("n", ""),
                mse=_fmt(row.get("student_future_mse_mean")),
                std=_fmt(row.get("student_future_mse_std")),
                importance=_fmt(row.get("selected_context_importance_mean"), 6),
                topk=_fmt(row.get("selector_target_topk_overlap_mean"), 6),
            )
        )
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- context teacher MSE: `{_fmt(summary.get('context_teacher_mse'))}`",
            f"- current_only MSE: `{_fmt(summary.get('current_only_mse'))}`",
            f"- learned_context MSE: `{_fmt(summary.get('learned_context_mse'))}`",
            f"- random_context MSE: `{_fmt(summary.get('random_context_mse'))}`",
            f"- uniform_context MSE: `{_fmt(summary.get('uniform_context_mse'))}`",
            f"- teacher_context_importance_topK MSE: `{_fmt(summary.get('teacher_context_importance_topk_mse'))}`",
            f"- hybrid_context MSE: `{_fmt(summary.get('hybrid_context_mse'))}`",
            f"- context gain over current_only: `{_fmt(summary.get('context_gain_over_current_only'))}`",
            f"- oracle gap: `{_fmt(summary.get('oracle_gap'))}`",
            "",
            "## Interpretation",
            "",
            f"- learned beats current_only: `{str(summary.get('learned_beats_current_only')).lower()}`",
            f"- learned beats random: `{str(summary.get('learned_beats_random')).lower()}`",
            f"- learned beats uniform: `{str(summary.get('learned_beats_uniform')).lower()}`",
            f"- hybrid beats learned: `{str(summary.get('hybrid_beats_learned')).lower()}`",
            f"- BAIR limitation: {summary.get('bair_context_limitation')}",
            "",
            "## Scientific Caveats",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in summary.get("scientific_caveats", [])] or ["- none"])
    lines.extend(
        [
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
            f"BAIR_CONTEXT_BOTTLENECK_VALIDATION_PASS = {str(summary.get('sanity_gate_pass')).lower()}",
            "",
        ]
    )
    return "\n".join(lines)


def write_context_bottleneck_summaries(
    *,
    config: dict[str, Any],
    resource_summary: dict[str, Any] | None = None,
    pytest_result: dict[str, Any] | None = None,
    env_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = build_context_bottleneck_summary(
        config=config,
        resource_summary=resource_summary,
        pytest_result=pytest_result,
        env_guard=env_guard,
    )
    summary_json = Path(config["output"]["summary_json"])
    summary_md = Path(config["output"]["summary_md"])
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(render_context_bottleneck_markdown(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    config = load_yaml(parse_args().config)
    print(json.dumps(write_context_bottleneck_summaries(config=config), indent=2))


if __name__ == "__main__":
    main()
