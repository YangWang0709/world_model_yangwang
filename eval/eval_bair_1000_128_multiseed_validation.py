"""Summarize Step 15 BAIR 1000/128 multi-seed validation outputs."""

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


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bair_1000_128_multiseed_validation.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


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


def _numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if _is_finite(row.get(field))]


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _fmt(value: Any, precision: int = 8) -> str:
    if value is None:
        return "n/a"
    if _is_finite(value):
        return f"{float(value):.{precision}f}"
    return str(value)


def _policy_label(policy: str) -> str:
    if policy == "learned_selector":
        return "learned_selector_weighted_mse_alpha2"
    return policy


def aggregate_rows_by_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_policy_label(str(row.get("policy"))), []).append(row)

    fields = [
        "student_future_mse",
        "student_teacher_ratio",
        "selector_target_topk_overlap",
        "selected_teacher_importance_mean",
        "selected_vs_random_importance_gap",
    ]
    policies = {
        policy: {field: _mean_std(_numeric_values(policy_rows, field)) for field in fields}
        for policy, policy_rows in sorted(grouped.items())
    }
    learned = policies.get("learned_selector_weighted_mse_alpha2", {})
    random = policies.get("random_k", {})
    uniform = policies.get("uniform_k", {})
    teacher_topk = policies.get("teacher_importance_topk", {})

    learned_mse = (learned.get("student_future_mse") or {}).get("mean")
    random_mse = (random.get("student_future_mse") or {}).get("mean")
    uniform_mse = (uniform.get("student_future_mse") or {}).get("mean")
    teacher_topk_mse = (teacher_topk.get("student_future_mse") or {}).get("mean")
    learned_importance = (learned.get("selected_teacher_importance_mean") or {}).get("mean")
    random_importance = (random.get("selected_teacher_importance_mean") or {}).get("mean")
    uniform_importance = (uniform.get("selected_teacher_importance_mean") or {}).get("mean")
    teacher_topk_importance = (teacher_topk.get("selected_teacher_importance_mean") or {}).get("mean")

    return {
        "policies": policies,
        "baseline_learned_mse_mean": learned_mse,
        "baseline_learned_mse_std": (learned.get("student_future_mse") or {}).get("std"),
        "baseline_random_mse_mean": random_mse,
        "baseline_random_mse_std": (random.get("student_future_mse") or {}).get("std"),
        "baseline_uniform_mse": uniform_mse,
        "baseline_teacher_importance_topk_mse": teacher_topk_mse,
        "baseline_learned_selected_importance_mean": learned_importance,
        "baseline_learned_selected_importance_std": (
            learned.get("selected_teacher_importance_mean") or {}
        ).get("std"),
        "baseline_random_selected_importance_mean": random_importance,
        "baseline_random_selected_importance_std": (
            random.get("selected_teacher_importance_mean") or {}
        ).get("std"),
        "baseline_uniform_selected_importance": uniform_importance,
        "baseline_teacher_importance_topk_selected_importance": teacher_topk_importance,
        "learned_vs_random_mse_delta": (
            float(learned_mse) - float(random_mse) if _is_finite(learned_mse) and _is_finite(random_mse) else None
        ),
        "learned_vs_uniform_mse_delta": (
            float(learned_mse) - float(uniform_mse) if _is_finite(learned_mse) and _is_finite(uniform_mse) else None
        ),
        "learned_vs_teacher_importance_topk_mse_delta": (
            float(learned_mse) - float(teacher_topk_mse)
            if _is_finite(learned_mse) and _is_finite(teacher_topk_mse)
            else None
        ),
        "learned_selected_importance_vs_random_delta": (
            float(learned_importance) - float(random_importance)
            if _is_finite(learned_importance) and _is_finite(random_importance)
            else None
        ),
        "learned_selected_importance_vs_uniform_delta": (
            float(learned_importance) - float(uniform_importance)
            if _is_finite(learned_importance) and _is_finite(uniform_importance)
            else None
        ),
        "learned_beats_random_mse": (
            float(learned_mse) < float(random_mse) if _is_finite(learned_mse) and _is_finite(random_mse) else False
        ),
        "learned_beats_uniform_mse": (
            float(learned_mse) <= float(uniform_mse) if _is_finite(learned_mse) and _is_finite(uniform_mse) else False
        ),
        "learned_selected_importance_beats_random": (
            float(learned_importance) > float(random_importance)
            if _is_finite(learned_importance) and _is_finite(random_importance)
            else False
        ),
        "learned_selected_importance_beats_uniform": (
            float(learned_importance) > float(uniform_importance)
            if _is_finite(learned_importance) and _is_finite(uniform_importance)
            else False
        ),
    }


def _write_rows_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def render_policy_aggregate_markdown(title: str, rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> str:
    lines = [
        f"# {title}",
        "",
        "| policy | n | student_mse_mean | student_mse_std | ratio_mean | selected_importance_mean | selected_importance_std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, stats in aggregate.get("policies", {}).items():
        lines.append(
            "| {policy} | {n} | {mse} | {mse_std} | {ratio} | {importance} | {importance_std} |".format(
                policy=policy,
                n=(stats.get("student_future_mse") or {}).get("n"),
                mse=_fmt((stats.get("student_future_mse") or {}).get("mean")),
                mse_std=_fmt((stats.get("student_future_mse") or {}).get("std")),
                ratio=_fmt((stats.get("student_teacher_ratio") or {}).get("mean"), 6),
                importance=_fmt((stats.get("selected_teacher_importance_mean") or {}).get("mean"), 6),
                importance_std=_fmt((stats.get("selected_teacher_importance_mean") or {}).get("std"), 6),
            )
        )
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            f"- learned vs random mean MSE delta: `{_fmt(aggregate.get('learned_vs_random_mse_delta'))}`",
            f"- learned vs uniform MSE delta: `{_fmt(aggregate.get('learned_vs_uniform_mse_delta'))}`",
            f"- learned vs teacher_importance_topk MSE delta: `{_fmt(aggregate.get('learned_vs_teacher_importance_topk_mse_delta'))}`",
            f"- learned selected importance vs random delta: `{_fmt(aggregate.get('learned_selected_importance_vs_random_delta'), 6)}`",
            f"- learned selected importance vs uniform delta: `{_fmt(aggregate.get('learned_selected_importance_vs_uniform_delta'), 6)}`",
            "",
            "## Rows",
            "",
            "| policy | seed | student_future_mse | teacher_mse | ratio | topK overlap | selected importance |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(rows, key=lambda item: (_policy_label(str(item.get("policy"))), int(item.get("seed", 0)))):
        lines.append(
            "| {policy} | {seed} | {student} | {teacher} | {ratio} | {topk} | {importance} |".format(
                policy=_policy_label(str(row.get("policy"))),
                seed=row.get("seed"),
                student=_fmt(row.get("student_future_mse")),
                teacher=_fmt(row.get("teacher_mse", row.get("teacher_future_mse"))),
                ratio=_fmt(row.get("student_teacher_ratio"), 6),
                topk=_fmt(row.get("selector_target_topk_overlap"), 6),
                importance=_fmt(row.get("selected_teacher_importance_mean"), 6),
            )
        )
    return "\n".join(lines) + "\n"


def write_baseline_multiseed_aggregate(
    rows: list[dict[str, Any]],
    run_dir: str | Path,
    output_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    output_cfg = output_cfg or {}
    aggregate = aggregate_rows_by_policy(rows)
    json_path = Path(output_cfg.get("aggregate_json", run_path / "baseline_multiseed_aggregate.json"))
    csv_path = Path(output_cfg.get("aggregate_csv", run_path / "baseline_multiseed_aggregate.csv"))
    md_path = Path(output_cfg.get("aggregate_md", run_path / "baseline_multiseed_aggregate.md"))
    payload = {"run_dir": str(run_path), "rows": rows, "aggregate": aggregate}
    _write_json(json_path, payload)
    aggregate_rows = []
    for policy, stats in aggregate["policies"].items():
        aggregate_rows.append(
            {
                "policy": policy,
                "n": (stats.get("student_future_mse") or {}).get("n"),
                "student_future_mse_mean": (stats.get("student_future_mse") or {}).get("mean"),
                "student_future_mse_std": (stats.get("student_future_mse") or {}).get("std"),
                "student_teacher_ratio_mean": (stats.get("student_teacher_ratio") or {}).get("mean"),
                "selected_teacher_importance_mean": (
                    stats.get("selected_teacher_importance_mean") or {}
                ).get("mean"),
                "selected_teacher_importance_std": (
                    stats.get("selected_teacher_importance_mean") or {}
                ).get("std"),
            }
        )
    _write_rows_csv(
        csv_path,
        aggregate_rows,
        [
            "policy",
            "n",
            "student_future_mse_mean",
            "student_future_mse_std",
            "student_teacher_ratio_mean",
            "selected_teacher_importance_mean",
            "selected_teacher_importance_std",
        ],
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_policy_aggregate_markdown("BAIR 1000/128 Baseline Multi-Seed Aggregate", rows, aggregate), encoding="utf-8")
    return {"baseline_multiseed_aggregate_json": str(json_path), "baseline_multiseed_aggregate_csv": str(csv_path), "baseline_multiseed_aggregate_md": str(md_path), "aggregate": aggregate}


def _aggregate_selector_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "test_importance_mse",
        "test_importance_mae",
        "test_pearson_corr_mean",
        "test_target_topk_overlap",
        "test_selected_teacher_importance_mean",
        "test_selected_vs_random_importance_gap",
    ]
    checks = {
        "all_success": all(bool(row.get("success", True)) for row in rows) and bool(rows),
        "all_metrics_finite": all(_is_finite(row.get("test_importance_mse")) for row in rows),
    }
    return {
        "num_seeds": len(rows),
        "seeds": [int(row.get("seed", -1)) for row in rows],
        **{field: _mean_std(_numeric_values(rows, field)) for field in fields},
        "sanity_gate": {"checks": checks, "pass": all(checks.values())},
    }


def _aggregate_student_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "student_future_mse",
        "student_teacher_ratio",
        "selector_target_topk_overlap",
        "selected_teacher_importance_mean",
        "selected_vs_random_importance_gap",
    ]
    checks = {
        "all_success": all(bool(row.get("success", True)) for row in rows) and bool(rows),
        "all_metrics_finite": all(_is_finite(row.get("student_future_mse")) for row in rows),
    }
    return {
        "num_seeds": len(rows),
        "seeds": [int(row.get("seed", -1)) for row in rows],
        **{field: _mean_std(_numeric_values(rows, field)) for field in fields},
        "sanity_gate": {"checks": checks, "pass": all(checks.values())},
    }


def _render_simple_multiseed_markdown(title: str, rows: list[dict[str, Any]], aggregate: dict[str, Any], columns: list[str]) -> str:
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |"]
    for row in sorted(rows, key=lambda item: int(item.get("seed", 0))):
        values = []
        for column in columns:
            value = row.get(column)
            values.append(_fmt(value, 6) if _is_finite(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Aggregate", "", "```json", json.dumps(aggregate, indent=2), "```", ""])
    return "\n".join(lines)


def write_selector_multiseed_summary(rows: list[dict[str, Any]], run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    aggregate = _aggregate_selector_rows(rows)
    json_path = run_path / "selector_multiseed_summary.json"
    csv_path = run_path / "selector_multiseed_summary.csv"
    md_path = run_path / "selector_multiseed_summary.md"
    payload = {"run_dir": str(run_path), "rows": rows, "aggregate": aggregate}
    _write_json(json_path, payload)
    columns = [
        "seed",
        "final_loss",
        "test_importance_mse",
        "test_pearson_corr_mean",
        "test_target_topk_overlap",
        "test_selected_teacher_importance_mean",
    ]
    _write_rows_csv(csv_path, rows, columns)
    md_path.write_text(_render_simple_multiseed_markdown("BAIR 1000/128 Selector Multi-Seed Summary", rows, aggregate, columns), encoding="utf-8")
    return {"selector_multiseed_summary_json": str(json_path), "selector_multiseed_summary_csv": str(csv_path), "selector_multiseed_summary_md": str(md_path), "aggregate": aggregate}


def write_student_multiseed_summary(rows: list[dict[str, Any]], run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    aggregate = _aggregate_student_rows(rows)
    json_path = run_path / "student_world_model_multiseed_summary.json"
    csv_path = run_path / "student_world_model_multiseed_summary.csv"
    md_path = run_path / "student_world_model_multiseed_summary.md"
    payload = {"run_dir": str(run_path), "rows": rows, "aggregate": aggregate}
    _write_json(json_path, payload)
    columns = [
        "seed",
        "final_loss",
        "student_future_mse",
        "teacher_mse",
        "student_teacher_ratio",
        "selector_target_topk_overlap",
        "selected_teacher_importance_mean",
    ]
    _write_rows_csv(csv_path, rows, columns)
    md_path.write_text(_render_simple_multiseed_markdown("BAIR 1000/128 StudentWorldModel Multi-Seed Summary", rows, aggregate, columns), encoding="utf-8")
    return {"student_multiseed_summary_json": str(json_path), "student_multiseed_summary_csv": str(csv_path), "student_multiseed_summary_md": str(md_path), "aggregate": aggregate}


def _token_shape(token_summary: dict[str, Any]) -> list[int] | None:
    train = token_summary.get("split_summaries", {}).get("train", {})
    shape = train.get("output_token_shape")
    return list(shape) if isinstance(shape, list) else None


def _split_count(summary: dict[str, Any], split: str) -> int:
    return int(summary.get("splits", {}).get(split, {}).get("exported", 0))


def _stat_mean(payload: dict[str, Any], key: str) -> float | None:
    value = (payload.get(key) or {}).get("mean")
    return float(value) if _is_finite(value) else None


def _stat_std(payload: dict[str, Any], key: str) -> float | None:
    value = (payload.get(key) or {}).get("std")
    return float(value) if _is_finite(value) else None


def load_step15_payloads(
    config: dict[str, Any],
    baseline_summary: str | Path | None = None,
    baseline_aggregate: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    run_root = Path(config["output"]["run_root"])
    teacher_dir = run_root / config["teacher"]["run_name"]
    selector_root = run_root / config["selector"]["run_root_name"]
    student_root = run_root / config["student_world_model"]["run_root_name"]
    baseline_dir = run_root / config["baseline"]["run_name"]
    step14_path = Path(config.get("step14_reference", {}).get("summary_json", ""))
    return {
        "subset": _read_json(Path(config["dataset"]["subset_dir"]) / "export_summary.json"),
        "tokens": _read_json(Path(config["tokens"]["output_root"]) / "extraction_summary.json"),
        "teacher": _read_json(teacher_dir / "summary.json"),
        "teacher_eval": _read_json(teacher_dir / "eval_summary.json"),
        "importance": _read_json(Path(config["importance"]["output_root"]) / "importance_summary.json"),
        "importance_eval": _read_json(Path(config["importance"]["output_root"]) / "eval_importance_summary.json"),
        "selector_multiseed": _read_json(selector_root / "selector_multiseed_summary.json"),
        "student_multiseed": _read_json(student_root / "student_world_model_multiseed_summary.json"),
        "baseline": _read_json(baseline_summary or baseline_dir / "baseline_summary.json"),
        "baseline_aggregate": _read_json(baseline_aggregate or baseline_dir / "baseline_multiseed_aggregate.json"),
        "step14": _read_json(step14_path, required=False) if str(step14_path) else {},
    }


def build_multiseed_validation_summary_from_payloads(
    *,
    config: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
    resource_summary: dict[str, Any] | None = None,
    pytest_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    subset = payloads.get("subset", {})
    tokens = payloads.get("tokens", {})
    teacher = payloads.get("teacher", {})
    teacher_eval = payloads.get("teacher_eval", {})
    importance_eval = payloads.get("importance_eval", {})
    selector_summary = payloads.get("selector_multiseed", {})
    student_summary = payloads.get("student_multiseed", {})
    baseline = payloads.get("baseline", {})
    baseline_aggregate_payload = payloads.get("baseline_aggregate", {})
    baseline_aggregate = baseline_aggregate_payload.get("aggregate") or aggregate_rows_by_policy(list(baseline.get("rows", [])))
    selector_aggregate = selector_summary.get("aggregate", {})
    student_aggregate = student_summary.get("aggregate", {})
    step14 = payloads.get("step14", {})

    train_samples = int(config["dataset"]["train_samples"])
    test_samples = int(config["dataset"]["test_samples"])
    topk = int(config["selector"]["topk"])
    token_shape = _token_shape(tokens)
    num_tokens = int(token_shape[1]) if token_shape and len(token_shape) == 3 else 392
    token_retention = float(topk) / float(num_tokens)
    teacher_mse = teacher_eval.get("eval_mse")
    learned_mse_mean = baseline_aggregate.get("baseline_learned_mse_mean")
    learned_importance_mean = baseline_aggregate.get("baseline_learned_selected_importance_mean")
    comparison_to_step14 = {
        "step14_train_samples": step14.get("train_samples"),
        "step14_test_samples": step14.get("test_samples"),
        "step14_baseline_learned_mse": step14.get("baseline_learned_mse"),
        "step14_baseline_random_mean_mse": step14.get("baseline_random_mean_mse"),
        "step14_baseline_uniform_mse": step14.get("baseline_uniform_mse"),
        "step15_learned_mse_mean_minus_step14_learned": (
            float(learned_mse_mean) - float(step14.get("baseline_learned_mse"))
            if _is_finite(learned_mse_mean) and _is_finite(step14.get("baseline_learned_mse"))
            else None
        ),
        "step15_random_mse_mean_minus_step14_random_mean": (
            float(baseline_aggregate.get("baseline_random_mse_mean")) - float(step14.get("baseline_random_mean_mse"))
            if _is_finite(baseline_aggregate.get("baseline_random_mse_mean")) and _is_finite(step14.get("baseline_random_mean_mse"))
            else None
        ),
        "step15_has_multiseed_learned_std": _is_finite(baseline_aggregate.get("baseline_learned_mse_std")),
    }
    checks = {
        "pipeline_outputs_present": all(
            bool(payloads.get(key))
            for key in (
                "subset",
                "tokens",
                "teacher",
                "teacher_eval",
                "importance",
                "importance_eval",
                "selector_multiseed",
                "student_multiseed",
                "baseline",
                "baseline_aggregate",
            )
        ),
        "train_subset_count_1000": _split_count(subset, "train") == train_samples,
        "test_subset_count_128": _split_count(subset, "test") == test_samples,
        "token_extraction_no_fallback": bool(tokens) and not bool(tokens.get("used_fallback", True)),
        "token_shape_expected": token_shape is not None and token_shape[1:] == [392, 768],
        "teacher_eval_mse_finite": _is_finite(teacher_mse),
        "importance_stats_finite": all(
            _is_finite(importance_eval.get(key))
            for key in (
                "importance_mean",
                "importance_std",
                "normalized_importance_mean",
                "normalized_importance_std",
                "base_loss_mean",
                "masked_loss_mean",
            )
        ),
        "selector_multiseed_metrics_finite": bool(selector_aggregate.get("sanity_gate", {}).get("pass", False)),
        "student_multiseed_metrics_finite": bool(student_aggregate.get("sanity_gate", {}).get("pass", False)),
        "baseline_multiseed_summary_exists": len(list(baseline.get("rows", []))) >= 8,
        "oom_false": not bool((resource_summary or {}).get("oom", False)),
    }
    if pytest_result is not None:
        checks["pytest_passed"] = bool(pytest_result.get("passed", False))

    caveats = []
    if not baseline_aggregate.get("learned_beats_random_mse", False):
        caveats.append("learned weighted_mse_alpha2 mean MSE did not beat random_k mean MSE")
    if not baseline_aggregate.get("learned_beats_uniform_mse", False):
        caveats.append("learned weighted_mse_alpha2 mean MSE did not beat uniform_k MSE")
    if not baseline_aggregate.get("learned_selected_importance_beats_random", False):
        caveats.append("learned weighted_mse_alpha2 selected importance did not beat random_k mean")
    if not baseline_aggregate.get("learned_selected_importance_beats_uniform", False):
        caveats.append("learned weighted_mse_alpha2 selected importance did not beat uniform_k")

    summary = {
        "stage": "bair_1000_128_multiseed_validation",
        "train_samples": train_samples,
        "test_samples": test_samples,
        "subset_path": config["dataset"]["subset_dir"],
        "token_output_root": config["tokens"]["output_root"],
        "importance_output_root": config["importance"]["output_root"],
        "token_shape": token_shape,
        "topk": topk,
        "token_retention_ratio": token_retention,
        "teacher_eval_mse": float(teacher_mse) if _is_finite(teacher_mse) else None,
        "selector_loss": "weighted_mse_alpha2",
        "selector_seeds": list(config["selector"]["seeds"]),
        "selector_test_importance_mse_mean": _stat_mean(selector_aggregate, "test_importance_mse"),
        "selector_test_importance_mse_std": _stat_std(selector_aggregate, "test_importance_mse"),
        "selector_test_target_topk_overlap_mean": _stat_mean(selector_aggregate, "test_target_topk_overlap"),
        "selector_test_target_topk_overlap_std": _stat_std(selector_aggregate, "test_target_topk_overlap"),
        "selector_selected_teacher_importance_mean": _stat_mean(selector_aggregate, "test_selected_teacher_importance_mean"),
        "selector_selected_teacher_importance_std": _stat_std(selector_aggregate, "test_selected_teacher_importance_mean"),
        "student_future_mse_mean": _stat_mean(student_aggregate, "student_future_mse"),
        "student_future_mse_std": _stat_std(student_aggregate, "student_future_mse"),
        "student_teacher_ratio_mean": _stat_mean(student_aggregate, "student_teacher_ratio"),
        "student_teacher_ratio_std": _stat_std(student_aggregate, "student_teacher_ratio"),
        **baseline_aggregate,
        "comparison_to_step14": comparison_to_step14,
        "importance_stats": {
            key: importance_eval.get(key)
            for key in (
                "importance_mean",
                "importance_std",
                "importance_min",
                "importance_max",
                "normalized_importance_mean",
                "normalized_importance_std",
                "normalized_importance_min",
                "normalized_importance_max",
                "base_loss_mean",
                "masked_loss_mean",
                "positive_importance_ratio",
            )
        },
        "teacher_summary": teacher,
        "selector_rows": selector_summary.get("rows", []),
        "selector_aggregate": selector_aggregate,
        "student_rows": student_summary.get("rows", []),
        "student_aggregate": student_aggregate,
        "baseline_rows": baseline.get("rows", []),
        "baseline_aggregate": baseline_aggregate,
        "sanity_gate": {"checks": checks, "pass": all(checks.values())},
        "sanity_gate_pass": all(checks.values()),
        "scientific_caveats": caveats,
        "resource_summary": resource_summary or {},
        "pytest_result": pytest_result,
        "cloud_required": False,
    }
    return summary


def render_multiseed_validation_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# BAIR 1000/128 Multi-Seed Validation Summary",
        "",
        f"- stage: `{summary['stage']}`",
        f"- train/test samples: `{summary['train_samples']}` / `{summary['test_samples']}`",
        f"- token shape: `{summary.get('token_shape')}`",
        f"- topK: `{summary['topk']}`",
        f"- token retention ratio: `{_fmt(summary['token_retention_ratio'], 12)}`",
        f"- Teacher eval MSE: `{_fmt(summary.get('teacher_eval_mse'))}`",
        f"- selector loss: `{summary.get('selector_loss')}`",
        f"- selector seeds: `{summary.get('selector_seeds')}`",
        f"- sanity gate pass: `{str(summary.get('sanity_gate_pass')).lower()}`",
        f"- cloud required: `{str(summary.get('cloud_required')).lower()}`",
        "",
        "## Selector Multi-Seed",
        "",
        "| seed | importance_mse | topK_overlap | selected_importance |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary.get("selector_rows", []), key=lambda item: int(item.get("seed", 0))):
        lines.append(
            f"| {row.get('seed')} | {_fmt(row.get('test_importance_mse'))} | {_fmt(row.get('test_target_topk_overlap'), 6)} | {_fmt(row.get('test_selected_teacher_importance_mean'), 6)} |"
        )
    lines.extend(
        [
            "",
            "## StudentWorldModel Multi-Seed",
            "",
            "| seed | student_future_mse | teacher_mse | ratio | selected_importance |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(summary.get("student_rows", []), key=lambda item: int(item.get("seed", 0))):
        lines.append(
            f"| {row.get('seed')} | {_fmt(row.get('student_future_mse'))} | {_fmt(row.get('teacher_mse'))} | {_fmt(row.get('student_teacher_ratio'), 6)} | {_fmt(row.get('selected_teacher_importance_mean'), 6)} |"
        )
    lines.extend(
        [
            "",
            "## Baseline Aggregate",
            "",
            "| policy | n | mse_mean | mse_std | selected_importance_mean | selected_importance_std |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy, stats in summary.get("baseline_aggregate", {}).get("policies", {}).items():
        lines.append(
            "| {policy} | {n} | {mse} | {mse_std} | {importance} | {importance_std} |".format(
                policy=policy,
                n=(stats.get("student_future_mse") or {}).get("n"),
                mse=_fmt((stats.get("student_future_mse") or {}).get("mean")),
                mse_std=_fmt((stats.get("student_future_mse") or {}).get("std")),
                importance=_fmt((stats.get("selected_teacher_importance_mean") or {}).get("mean"), 6),
                importance_std=_fmt((stats.get("selected_teacher_importance_mean") or {}).get("std"), 6),
            )
        )
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            f"- learned mean vs random mean MSE delta: `{_fmt(summary.get('learned_vs_random_mse_delta'))}`",
            f"- learned mean vs uniform MSE delta: `{_fmt(summary.get('learned_vs_uniform_mse_delta'))}`",
            f"- learned mean vs teacher_importance_topk MSE delta: `{_fmt(summary.get('learned_vs_teacher_importance_topk_mse_delta'))}`",
            f"- learned selected importance vs random delta: `{_fmt(summary.get('learned_selected_importance_vs_random_delta'), 6)}`",
            f"- learned selected importance vs uniform delta: `{_fmt(summary.get('learned_selected_importance_vs_uniform_delta'), 6)}`",
            "",
            "## Comparison To Step 14",
            "",
            "```json",
            json.dumps(summary.get("comparison_to_step14", {}), indent=2),
            "```",
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
            f"BAIR_1000_128_MULTI_SEED_VALIDATION_PASS = {str(summary.get('sanity_gate_pass')).lower()}",
            "",
        ]
    )
    return "\n".join(lines)


def write_multiseed_validation_summary(
    config: dict[str, Any],
    run_root: str | Path | None = None,
    baseline_summary: str | Path | None = None,
    baseline_aggregate: str | Path | None = None,
    resource_summary: dict[str, Any] | None = None,
    pytest_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if run_root is not None:
        config = json.loads(json.dumps(config))
        config["output"]["summary_json"] = str(Path(run_root) / "multiseed_validation_summary.json")
        config["output"]["summary_md"] = str(Path(run_root) / "multiseed_validation_summary.md")
    previous = _read_json(config["output"]["summary_json"], required=False)
    if resource_summary is None:
        resource_summary = previous.get("resource_summary", {})
    if pytest_result is None:
        pytest_result = previous.get("pytest_result")
    payloads = load_step15_payloads(config, baseline_summary=baseline_summary, baseline_aggregate=baseline_aggregate)
    summary = build_multiseed_validation_summary_from_payloads(
        config=config,
        payloads=payloads,
        resource_summary=resource_summary,
        pytest_result=pytest_result,
    )
    json_path = Path(config["output"]["summary_json"])
    md_path = Path(config["output"]["summary_md"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(render_multiseed_validation_markdown(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-root")
    parser.add_argument("--baseline-summary")
    parser.add_argument("--baseline-aggregate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_multiseed_validation_summary(
        load_yaml(args.config),
        run_root=args.run_root,
        baseline_summary=args.baseline_summary,
        baseline_aggregate=args.baseline_aggregate,
    )
    print("BAIR_1000_128_MULTI_SEED_VALIDATION_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
