"""Summarize Step 16 BAIR downstream utilization ablation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "downstream_utilization_bair_1000_128.yaml"
PHASE_COLUMNS = [
    "phase",
    "variant",
    "seed",
    "success",
    "student_future_mse",
    "teacher_mse",
    "student_teacher_ratio",
    "selector_target_topk_overlap",
    "selected_teacher_importance_mean",
    "selected_vs_random_importance_gap",
    "error",
]
PHASE_B_COLUMNS = [
    "variant",
    "seeds",
    "n",
    "student_future_mse_mean",
    "student_future_mse_std",
    "student_teacher_ratio_mean",
    "student_teacher_ratio_std",
    "selected_teacher_importance_mean",
    "selected_teacher_importance_std",
    "selector_target_topk_overlap_mean",
    "selector_target_topk_overlap_std",
]


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
    json_path = Path(path)
    if not json_path.exists():
        if required:
            raise FileNotFoundError(json_path)
        return {}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {json_path}")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fmt(value: Any, precision: int = 8) -> str:
    if value is None:
        return "n/a"
    if _is_finite(value):
        return f"{float(value):.{precision}f}"
    return str(value)


def _numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if bool(row.get("success", True)) and _is_finite(row.get(field))]


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _load_phase_rows(run_dir: str | Path, phase: str) -> list[dict[str, Any]]:
    phase_dir = Path(run_dir) / phase
    if not phase_dir.exists():
        return []
    rows = []
    for summary_path in sorted(phase_dir.glob("*/summary.json")):
        row = _read_json(summary_path)
        row.setdefault("summary_path", str(summary_path))
        rows.append(row)
    return rows


def _best_row(rows: list[dict[str, Any]], metric: str = "student_future_mse") -> dict[str, Any] | None:
    successful = [row for row in rows if bool(row.get("success", True)) and _is_finite(row.get(metric))]
    if not successful:
        return None
    return min(successful, key=lambda row: float(row[metric]))


def aggregate_phase_b_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not bool(row.get("success", True)):
            continue
        grouped.setdefault(str(row.get("variant")), []).append(row)
    aggregate_rows = []
    for variant, variant_rows in sorted(grouped.items()):
        mse = _mean_std(_numeric_values(variant_rows, "student_future_mse"))
        ratio = _mean_std(_numeric_values(variant_rows, "student_teacher_ratio"))
        importance = _mean_std(_numeric_values(variant_rows, "selected_teacher_importance_mean"))
        topk = _mean_std(_numeric_values(variant_rows, "selector_target_topk_overlap"))
        aggregate_rows.append(
            {
                "variant": variant,
                "seeds": ",".join(str(int(row.get("seed", -1))) for row in sorted(variant_rows, key=lambda item: int(item.get("seed", 0)))),
                "n": mse["n"],
                "student_future_mse_mean": mse["mean"],
                "student_future_mse_std": mse["std"],
                "student_teacher_ratio_mean": ratio["mean"],
                "student_teacher_ratio_std": ratio["std"],
                "selected_teacher_importance_mean": importance["mean"],
                "selected_teacher_importance_std": importance["std"],
                "selector_target_topk_overlap_mean": topk["mean"],
                "selector_target_topk_overlap_std": topk["std"],
            }
        )
    return aggregate_rows


def render_phase_a_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Step16 Phase A Summary",
        "",
        "| variant | seed | success | student_future_mse | ratio | selected_importance | topK_overlap |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: str(item.get("variant"))):
        lines.append(
            "| {variant} | {seed} | {success} | {mse} | {ratio} | {importance} | {topk} |".format(
                variant=row.get("variant"),
                seed=row.get("seed"),
                success=str(bool(row.get("success", True))).lower(),
                mse=_fmt(row.get("student_future_mse")),
                ratio=_fmt(row.get("student_teacher_ratio"), 6),
                importance=_fmt(row.get("selected_teacher_importance_mean"), 6),
                topk=_fmt(row.get("selector_target_topk_overlap"), 6),
            )
        )
    return "\n".join(lines) + "\n"


def render_phase_b_markdown(rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Step16 Phase B Summary",
        "",
        "## Aggregate",
        "",
        "| variant | seeds | n | mse_mean | mse_std | ratio_mean | ratio_std | selected_importance_mean | selected_importance_std | topK_overlap_mean | topK_overlap_std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(aggregate_rows, key=lambda item: float(item["student_future_mse_mean"]) if _is_finite(item.get("student_future_mse_mean")) else 1e9):
        lines.append(
            "| {variant} | {seeds} | {n} | {mse} | {mse_std} | {ratio} | {ratio_std} | {importance} | {importance_std} | {topk} | {topk_std} |".format(
                variant=row.get("variant"),
                seeds=row.get("seeds"),
                n=row.get("n"),
                mse=_fmt(row.get("student_future_mse_mean")),
                mse_std=_fmt(row.get("student_future_mse_std")),
                ratio=_fmt(row.get("student_teacher_ratio_mean"), 6),
                ratio_std=_fmt(row.get("student_teacher_ratio_std"), 6),
                importance=_fmt(row.get("selected_teacher_importance_mean"), 6),
                importance_std=_fmt(row.get("selected_teacher_importance_std"), 6),
                topk=_fmt(row.get("selector_target_topk_overlap_mean"), 6),
                topk_std=_fmt(row.get("selector_target_topk_overlap_std"), 6),
            )
        )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| variant | seed | student_future_mse | ratio | selected_importance | topK_overlap |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (str(item.get("variant")), int(item.get("seed", 0)))):
        lines.append(
            "| {variant} | {seed} | {mse} | {ratio} | {importance} | {topk} |".format(
                variant=row.get("variant"),
                seed=row.get("seed"),
                mse=_fmt(row.get("student_future_mse")),
                ratio=_fmt(row.get("student_teacher_ratio"), 6),
                importance=_fmt(row.get("selected_teacher_importance_mean"), 6),
                topk=_fmt(row.get("selector_target_topk_overlap"), 6),
            )
        )
    return "\n".join(lines) + "\n"


def _step15_reference(config: dict[str, Any], step15_aggregate_path: str | Path | None) -> dict[str, Any]:
    ref = dict(config.get("step15_reference", {}))
    aggregate_path = step15_aggregate_path or ref.get("baseline_aggregate")
    aggregate_payload = _read_json(aggregate_path, required=False) if aggregate_path else {}
    aggregate = aggregate_payload.get("aggregate", {})
    return {
        "step15_learned_mse_mean": float(ref.get("learned_mse_mean", aggregate.get("baseline_learned_mse_mean"))),
        "step15_random_mse_mean": float(ref.get("random_mse_mean", aggregate.get("baseline_random_mse_mean"))),
        "step15_uniform_mse": float(ref.get("uniform_mse", aggregate.get("baseline_uniform_mse"))),
        "step15_teacher_importance_topk_mse": float(
            ref.get("teacher_importance_topk_mse", aggregate.get("baseline_teacher_importance_topk_mse"))
        ),
        "step15_learned_selected_importance_mean": float(
            ref.get("learned_selected_importance_mean", aggregate.get("baseline_learned_selected_importance_mean", 0.0))
        ),
        "baseline_aggregate_path": str(aggregate_path) if aggregate_path else None,
    }


def build_downstream_summary(
    *,
    config: dict[str, Any],
    run_dir: str | Path,
    phase_a_rows: list[dict[str, Any]],
    phase_b_rows: list[dict[str, Any]],
    resource_summary: dict[str, Any] | None = None,
    pytest_result: dict[str, Any] | None = None,
    env_guard: dict[str, Any] | None = None,
    step15_aggregate_path: str | Path | None = None,
) -> dict[str, Any]:
    phase_b_aggregate = aggregate_phase_b_rows(phase_b_rows)
    phase_a_best = _best_row(phase_a_rows)
    phase_b_best = None
    if phase_b_aggregate:
        phase_b_best = min(
            [row for row in phase_b_aggregate if _is_finite(row.get("student_future_mse_mean"))],
            key=lambda row: float(row["student_future_mse_mean"]),
            default=None,
        )
    refs = _step15_reference(config, step15_aggregate_path)
    best_mse = phase_b_best.get("student_future_mse_mean") if phase_b_best else None
    best_importance = phase_b_best.get("selected_teacher_importance_mean") if phase_b_best else None
    best_topk = phase_b_best.get("selector_target_topk_overlap_mean") if phase_b_best else None
    phase_a_success = [row for row in phase_a_rows if bool(row.get("success", True))]
    phase_b_success = [row for row in phase_b_rows if bool(row.get("success", True))]
    phase_b_complete_variants = [row for row in phase_b_aggregate if int(row.get("n", 0) or 0) >= 3]

    best_beats_learned = bool(_is_finite(best_mse) and float(best_mse) < refs["step15_learned_mse_mean"])
    best_beats_uniform = bool(_is_finite(best_mse) and float(best_mse) <= refs["step15_uniform_mse"])
    best_beats_random = bool(_is_finite(best_mse) and float(best_mse) < refs["step15_random_mse_mean"])
    caveats = []
    if not best_beats_learned:
        caveats.append("best Step16 variant did not beat Step15 learned weighted_mse_alpha2 MSE mean")
    if not best_beats_uniform:
        caveats.append("best Step16 variant did not beat Step15 uniform_k MSE")
    if _is_finite(best_importance) and float(best_importance) < refs["step15_learned_selected_importance_mean"]:
        caveats.append("best Step16 selected importance is below Step15 learned selected importance mean")
    if not any(str(row.get("variant", "")).startswith("hybrid_") for row in phase_b_aggregate):
        caveats.append("no hybrid variant reached Phase B top-2")
    if best_mse is not None and not best_beats_uniform:
        caveats.append("downstream architecture alone was insufficient to beat uniform; Step17 should prioritize action-conditioned world model")

    checks = {
        "phase_a_at_least_6_success": len(phase_a_success) >= 6,
        "phase_b_at_least_2_variants": len(phase_b_complete_variants) >= 2,
        "all_success_metrics_finite": all(
            _is_finite(row.get("student_future_mse"))
            and _is_finite(row.get("teacher_mse"))
            and _is_finite(row.get("student_teacher_ratio"))
            for row in phase_a_success + phase_b_success
        ),
        "oom_false": not bool((resource_summary or {}).get("oom", False)),
        "pytest_passed": bool((pytest_result or {}).get("passed", True)),
        "env_isaaclab_unpolluted": not bool((env_guard or {}).get("tensorflow", False))
        and not bool((env_guard or {}).get("tensorflow_datasets", False)),
    }
    summary = {
        "stage": "downstream_utilization_bair_1000_128",
        "run_dir": str(run_dir),
        "train_samples": int(config["data"]["max_train_samples"]),
        "test_samples": int(config["data"]["max_test_samples"]),
        "topk": int(config["selection"]["topk"]),
        "token_retention_ratio": float(config["selection"]["token_retention_ratio"]),
        "input_paths": {
            "train_token_shard_dir": config["data"]["train_token_shard_dir"],
            "test_token_shard_dir": config["data"]["test_token_shard_dir"],
            "train_importance_shard_dir": config["data"]["train_importance_shard_dir"],
            "test_importance_shard_dir": config["data"]["test_importance_shard_dir"],
            "teacher_checkpoint": config["teacher_reference"]["checkpoint"],
            "selector_root": config["learned_selectors"]["root"],
            "baseline_aggregate": refs["baseline_aggregate_path"],
        },
        "phase_a_variants": phase_a_rows,
        "phase_a_success_count": len(phase_a_success),
        "phase_a_best_variant": phase_a_best.get("variant") if phase_a_best else None,
        "phase_b_rows": phase_b_rows,
        "phase_b_variants": phase_b_aggregate,
        "phase_b_best_mse_variant": phase_b_best.get("variant") if phase_b_best else None,
        "phase_b_best_mse_mean": float(best_mse) if _is_finite(best_mse) else None,
        "phase_b_best_mse_std": phase_b_best.get("student_future_mse_std") if phase_b_best else None,
        **refs,
        "best_beats_step15_learned": best_beats_learned,
        "best_beats_step15_uniform": best_beats_uniform,
        "best_beats_step15_random_mean": best_beats_random,
        "gap_to_teacher_importance_topk": (
            float(best_mse) - refs["step15_teacher_importance_topk_mse"] if _is_finite(best_mse) else None
        ),
        "best_selected_importance_mean": float(best_importance) if _is_finite(best_importance) else None,
        "best_topk_overlap_mean": float(best_topk) if _is_finite(best_topk) else None,
        "hybrid_learned_uniform_reached_phase_b": any(str(row.get("variant", "")).startswith("hybrid_") for row in phase_b_aggregate),
        "cross_attention_reached_phase_b": any("cross_attention" in str(row.get("variant", "")) for row in phase_b_aggregate),
        "scientific_caveats": caveats,
        "sanity_gate": {"checks": checks, "pass": all(checks.values())},
        "sanity_gate_pass": all(checks.values()),
        "resource_summary": resource_summary or {},
        "pytest_result": pytest_result or {},
        "env_guard": env_guard or {},
        "cloud_required": False,
        "failed_variants": [row for row in phase_a_rows + phase_b_rows if not bool(row.get("success", True))],
    }
    return summary


def render_downstream_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# BAIR Downstream Utilization Ablation Summary",
        "",
        f"- stage: `{summary['stage']}`",
        f"- train/test samples: `{summary['train_samples']}` / `{summary['test_samples']}`",
        f"- topK: `{summary['topk']}`",
        f"- token retention ratio: `{_fmt(summary['token_retention_ratio'], 12)}`",
        f"- phase A best: `{summary.get('phase_a_best_variant')}`",
        f"- phase B best: `{summary.get('phase_b_best_mse_variant')}`",
        f"- phase B best MSE mean/std: `{_fmt(summary.get('phase_b_best_mse_mean'))}` / `{_fmt(summary.get('phase_b_best_mse_std'))}`",
        f"- sanity gate pass: `{str(summary.get('sanity_gate_pass')).lower()}`",
        f"- cloud required: `{str(summary.get('cloud_required')).lower()}`",
        "",
        "## Phase A",
        "",
        "| variant | mse | ratio | selected_importance | topK_overlap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary.get("phase_a_variants", []), key=lambda item: float(item["student_future_mse"]) if _is_finite(item.get("student_future_mse")) else 1e9):
        lines.append(
            "| {variant} | {mse} | {ratio} | {importance} | {topk} |".format(
                variant=row.get("variant"),
                mse=_fmt(row.get("student_future_mse")),
                ratio=_fmt(row.get("student_teacher_ratio"), 6),
                importance=_fmt(row.get("selected_teacher_importance_mean"), 6),
                topk=_fmt(row.get("selector_target_topk_overlap"), 6),
            )
        )
    lines.extend(
        [
            "",
            "## Phase B",
            "",
            "| variant | seeds | mse_mean | mse_std | ratio_mean | selected_importance_mean | topK_overlap_mean |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(summary.get("phase_b_variants", []), key=lambda item: float(item["student_future_mse_mean"]) if _is_finite(item.get("student_future_mse_mean")) else 1e9):
        lines.append(
            "| {variant} | {seeds} | {mse} | {mse_std} | {ratio} | {importance} | {topk} |".format(
                variant=row.get("variant"),
                seeds=row.get("seeds"),
                mse=_fmt(row.get("student_future_mse_mean")),
                mse_std=_fmt(row.get("student_future_mse_std")),
                ratio=_fmt(row.get("student_teacher_ratio_mean"), 6),
                importance=_fmt(row.get("selected_teacher_importance_mean"), 6),
                topk=_fmt(row.get("selector_target_topk_overlap_mean"), 6),
            )
        )
    lines.extend(
        [
            "",
            "## Step15 Comparison",
            "",
            f"- best vs Step15 learned MSE delta: `{_fmt((summary.get('phase_b_best_mse_mean') or 0) - summary.get('step15_learned_mse_mean')) if _is_finite(summary.get('phase_b_best_mse_mean')) else 'n/a'}`",
            f"- best vs Step15 random mean MSE delta: `{_fmt((summary.get('phase_b_best_mse_mean') or 0) - summary.get('step15_random_mse_mean')) if _is_finite(summary.get('phase_b_best_mse_mean')) else 'n/a'}`",
            f"- best vs Step15 uniform MSE delta: `{_fmt((summary.get('phase_b_best_mse_mean') or 0) - summary.get('step15_uniform_mse')) if _is_finite(summary.get('phase_b_best_mse_mean')) else 'n/a'}`",
            f"- best vs teacher_importance_topk MSE delta: `{_fmt(summary.get('gap_to_teacher_importance_topk'))}`",
            f"- best beats Step15 learned: `{str(summary.get('best_beats_step15_learned')).lower()}`",
            f"- best beats Step15 random mean: `{str(summary.get('best_beats_step15_random_mean')).lower()}`",
            f"- best beats Step15 uniform: `{str(summary.get('best_beats_step15_uniform')).lower()}`",
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
            f"BAIR_DOWNSTREAM_UTILIZATION_ABLATION_PASS = {str(summary.get('sanity_gate_pass')).lower()}",
            "",
        ]
    )
    return "\n".join(lines)


def write_downstream_utilization_summaries(
    *,
    run_dir: str | Path,
    config: dict[str, Any],
    step15_aggregate_path: str | Path | None = None,
    resource_summary: dict[str, Any] | None = None,
    pytest_result: dict[str, Any] | None = None,
    env_guard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    phase_a_rows = _load_phase_rows(run_path, "phase_a")
    phase_b_rows = _load_phase_rows(run_path, "phase_b")
    phase_b_aggregate = aggregate_phase_b_rows(phase_b_rows)
    summary = build_downstream_summary(
        config=config,
        run_dir=run_path,
        phase_a_rows=phase_a_rows,
        phase_b_rows=phase_b_rows,
        resource_summary=resource_summary,
        pytest_result=pytest_result,
        env_guard=env_guard,
        step15_aggregate_path=step15_aggregate_path,
    )
    _write_json(run_path / "phase_a_summary.json", {"rows": phase_a_rows})
    _write_csv(run_path / "phase_a_summary.csv", phase_a_rows, PHASE_COLUMNS)
    (run_path / "phase_a_summary.md").write_text(render_phase_a_markdown(phase_a_rows), encoding="utf-8")
    _write_json(run_path / "phase_b_summary.json", {"rows": phase_b_rows, "aggregate": phase_b_aggregate})
    _write_csv(run_path / "phase_b_summary.csv", phase_b_aggregate, PHASE_B_COLUMNS)
    (run_path / "phase_b_summary.md").write_text(render_phase_b_markdown(phase_b_rows, phase_b_aggregate), encoding="utf-8")
    _write_json(run_path / "downstream_utilization_summary.json", summary)
    (run_path / "downstream_utilization_summary.md").write_text(render_downstream_markdown(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--step15-aggregate", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    run_dir = args.run_dir or Path(config["output"]["run_root"]) / config["output"]["run_name"]
    summary = write_downstream_utilization_summaries(
        run_dir=run_dir,
        config=config,
        step15_aggregate_path=args.step15_aggregate,
    )
    print("BAIR_DOWNSTREAM_UTILIZATION_ABLATION_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
