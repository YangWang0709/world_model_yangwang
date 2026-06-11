"""Summarize Step 14 BAIR 500/64 scale validation outputs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bair_500_64_scale_validation.yaml"


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


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _row_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return float(value) if _is_finite(value) else None


def compute_baseline_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    random_rows = [row for row in rows if row.get("policy") == "random_k"]
    learned = next((row for row in rows if row.get("policy") == "learned_selector"), None)
    uniform = next((row for row in rows if row.get("policy") == "uniform_k"), None)
    teacher_topk = next((row for row in rows if row.get("policy") == "teacher_importance_topk"), None)

    random_mse = _mean([float(row["student_future_mse"]) for row in random_rows if _is_finite(row.get("student_future_mse"))])
    random_importance = _mean(
        [
            float(row["selected_teacher_importance_mean"])
            for row in random_rows
            if _is_finite(row.get("selected_teacher_importance_mean"))
        ]
    )
    learned_mse = _row_value(learned or {}, "student_future_mse")
    learned_importance = _row_value(learned or {}, "selected_teacher_importance_mean")
    uniform_mse = _row_value(uniform or {}, "student_future_mse")
    uniform_importance = _row_value(uniform or {}, "selected_teacher_importance_mean")
    teacher_topk_mse = _row_value(teacher_topk or {}, "student_future_mse")
    teacher_topk_importance = _row_value(teacher_topk or {}, "selected_teacher_importance_mean")

    return {
        "baseline_learned_mse": learned_mse,
        "baseline_random_mean_mse": random_mse,
        "baseline_uniform_mse": uniform_mse,
        "baseline_teacher_importance_topk_mse": teacher_topk_mse,
        "baseline_learned_selected_importance": learned_importance,
        "baseline_random_mean_selected_importance": random_importance,
        "baseline_uniform_selected_importance": uniform_importance,
        "baseline_teacher_importance_topk_selected_importance": teacher_topk_importance,
        "learned_vs_random_mse_delta": (
            learned_mse - random_mse if learned_mse is not None and random_mse is not None else None
        ),
        "learned_vs_uniform_mse_delta": (
            learned_mse - uniform_mse if learned_mse is not None and uniform_mse is not None else None
        ),
        "learned_vs_teacher_importance_topk_mse_delta": (
            learned_mse - teacher_topk_mse
            if learned_mse is not None and teacher_topk_mse is not None
            else None
        ),
        "learned_selected_importance_vs_random_delta": (
            learned_importance - random_importance
            if learned_importance is not None and random_importance is not None
            else None
        ),
        "learned_selected_importance_vs_uniform_delta": (
            learned_importance - uniform_importance
            if learned_importance is not None and uniform_importance is not None
            else None
        ),
        "learned_beats_random_mse": (
            learned_mse < random_mse if learned_mse is not None and random_mse is not None else False
        ),
        "learned_beats_uniform_mse": (
            learned_mse <= uniform_mse if learned_mse is not None and uniform_mse is not None else False
        ),
        "learned_selected_importance_beats_random": (
            learned_importance > random_importance
            if learned_importance is not None and random_importance is not None
            else False
        ),
        "learned_selected_importance_beats_uniform": (
            learned_importance > uniform_importance
            if learned_importance is not None and uniform_importance is not None
            else False
        ),
    }


def _token_shape(token_summary: dict[str, Any]) -> list[int] | None:
    train = token_summary.get("split_summaries", {}).get("train", {})
    shape = train.get("output_token_shape")
    return list(shape) if isinstance(shape, list) else None


def _split_count(summary: dict[str, Any], split: str) -> int:
    return int(summary.get("splits", {}).get(split, {}).get("exported", 0))


def _importance_eval_stats(importance_eval: dict[str, Any]) -> dict[str, Any]:
    keys = (
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
    return {key: importance_eval.get(key) for key in keys}


def build_scale_validation_summary_from_payloads(
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
    importance = payloads.get("importance", {})
    importance_eval = payloads.get("importance_eval", {})
    selector = payloads.get("selector", {})
    student = payloads.get("student", {})
    student_eval = payloads.get("student_eval", {})
    gap = payloads.get("gap", {})
    baseline = payloads.get("baseline", {})
    baseline_rows = list(baseline.get("rows", []))
    comparisons = compute_baseline_comparisons(baseline_rows)

    train_samples = int(config["dataset"]["train_samples"])
    test_samples = int(config["dataset"]["test_samples"])
    topk = int(config["selector"]["topk"])
    token_shape = _token_shape(tokens)
    num_tokens = int(token_shape[1]) if token_shape and len(token_shape) == 3 else int(selector.get("num_tokens", 392))
    token_retention = float(topk) / float(num_tokens)
    student_mse = gap.get("student_future_mse", student_eval.get("student_future_mse", student.get("student_future_mse")))
    teacher_mse = gap.get("teacher_mse", teacher_eval.get("eval_mse"))
    selector_mse = selector.get("test_importance_mse")
    selector_corr = selector.get("test_pearson_corr_mean")
    selector_topk = selector.get("test_target_topk_overlap")
    selector_importance = selector.get("test_selected_teacher_importance_mean")
    checks = {
        "pipeline_outputs_present": all(bool(payloads.get(key)) for key in (
            "subset",
            "tokens",
            "teacher",
            "teacher_eval",
            "importance",
            "importance_eval",
            "selector",
            "student",
            "student_eval",
            "gap",
            "baseline",
        )),
        "train_subset_count_500": _split_count(subset, "train") == train_samples,
        "test_subset_count_64": _split_count(subset, "test") == test_samples,
        "token_extraction_no_fallback": bool(tokens) and not bool(tokens.get("used_fallback", True)),
        "token_shape_expected": token_shape is not None and token_shape[1:] == [392, 768],
        "teacher_eval_mse_finite": _is_finite(teacher_mse),
        "importance_stats_finite": all(_is_finite(importance_eval.get(key)) for key in (
            "importance_mean",
            "importance_std",
            "normalized_importance_mean",
            "normalized_importance_std",
            "base_loss_mean",
            "masked_loss_mean",
        )),
        "selector_metrics_finite": all(_is_finite(value) for value in (selector_mse, selector_corr, selector_topk, selector_importance)),
        "student_metrics_finite": all(_is_finite(value) for value in (student_mse, teacher_mse, gap.get("student_teacher_ratio"))),
        "baseline_summary_exists": bool(baseline_rows),
        "oom_false": not bool((resource_summary or {}).get("oom", False)),
    }
    if pytest_result is not None:
        checks["pytest_passed"] = bool(pytest_result.get("passed", False))

    advisory = {
        "learned_beats_random_mse": comparisons["learned_beats_random_mse"],
        "learned_selected_importance_beats_random": comparisons["learned_selected_importance_beats_random"],
        "learned_beats_uniform_mse": comparisons["learned_beats_uniform_mse"],
        "learned_close_to_teacher_importance_topk": (
            comparisons["learned_vs_teacher_importance_topk_mse_delta"] is not None
            and abs(float(comparisons["learned_vs_teacher_importance_topk_mse_delta"])) <= 0.15
        ),
    }
    caveats = []
    if not advisory["learned_beats_random_mse"]:
        caveats.append("learned weighted_mse did not beat random_k mean MSE in this 500/64 run")
    if not advisory["learned_beats_uniform_mse"]:
        caveats.append("learned weighted_mse did not beat uniform_k MSE in this 500/64 run")
    if not advisory["learned_selected_importance_beats_random"]:
        caveats.append("learned weighted_mse did not select higher-importance tokens than random_k mean")

    summary = {
        "stage": "bair_500_64_scale_validation",
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
        "selector_test_importance_mse": float(selector_mse) if _is_finite(selector_mse) else None,
        "selector_test_pearson_corr_mean": float(selector_corr) if _is_finite(selector_corr) else None,
        "selector_test_target_topk_overlap": float(selector_topk) if _is_finite(selector_topk) else None,
        "selector_selected_teacher_importance_mean": (
            float(selector_importance) if _is_finite(selector_importance) else None
        ),
        "student_future_mse": float(student_mse) if _is_finite(student_mse) else None,
        "teacher_mse": float(teacher_mse) if _is_finite(teacher_mse) else None,
        "student_teacher_ratio": (
            float(gap["student_teacher_ratio"]) if _is_finite(gap.get("student_teacher_ratio")) else None
        ),
        **comparisons,
        "importance_stats": _importance_eval_stats(importance_eval),
        "teacher_summary": teacher,
        "student_summary": student,
        "baseline_rows": baseline_rows,
        "sanity_gate": {"checks": checks, "pass": all(checks.values())},
        "sanity_gate_pass": all(checks.values()),
        "advisory_metrics": advisory,
        "caveats": caveats,
        "resource_summary": resource_summary or {},
        "pytest_result": pytest_result,
        "cloud_required": False,
    }
    return summary


def _run_dir(root: str | Path, run_name: str) -> Path:
    return Path(root) / run_name


def load_step14_payloads(config: dict[str, Any], baseline_summary: str | Path | None = None) -> dict[str, dict[str, Any]]:
    run_root = Path(config["output"]["run_root"])
    teacher_dir = _run_dir(run_root, config["teacher"]["run_name"])
    selector_dir = _run_dir(run_root, config["selector"]["run_name"])
    student_dir = _run_dir(run_root, config["student_world_model"]["run_name"])
    baseline_path = Path(baseline_summary or _run_dir(run_root, config["baseline"]["run_name"]) / "baseline_summary.json")
    return {
        "subset": _read_json(Path(config["dataset"]["subset_dir"]) / "export_summary.json"),
        "tokens": _read_json(Path(config["tokens"]["output_root"]) / "extraction_summary.json"),
        "teacher": _read_json(teacher_dir / "summary.json"),
        "teacher_eval": _read_json(teacher_dir / "eval_summary.json"),
        "importance": _read_json(Path(config["importance"]["output_root"]) / "importance_summary.json"),
        "importance_eval": _read_json(Path(config["importance"]["output_root"]) / "eval_importance_summary.json"),
        "selector": _read_json(selector_dir / "summary.json"),
        "student": _read_json(student_dir / "summary.json"),
        "student_eval": _read_json(student_dir / "eval_summary.json"),
        "gap": _read_json(student_dir / "teacher_student_gap_summary.json"),
        "baseline": _read_json(baseline_path),
    }


def _fmt(value: Any, precision: int = 8) -> str:
    if value is None:
        return "n/a"
    if _is_finite(value):
        return f"{float(value):.{precision}f}"
    return str(value)


def render_scale_validation_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# BAIR 500/64 Scale Validation Summary",
        "",
        f"- stage: `{summary['stage']}`",
        f"- train/test samples: `{summary['train_samples']}` / `{summary['test_samples']}`",
        f"- token shape: `{summary.get('token_shape')}`",
        f"- topK: `{summary['topk']}`",
        f"- token retention ratio: `{_fmt(summary['token_retention_ratio'], 12)}`",
        f"- Teacher eval MSE: `{_fmt(summary.get('teacher_eval_mse'))}`",
        f"- Student future MSE: `{_fmt(summary.get('student_future_mse'))}`",
        f"- Student/Teacher ratio: `{_fmt(summary.get('student_teacher_ratio'), 6)}`",
        f"- sanity gate pass: `{str(summary.get('sanity_gate_pass')).lower()}`",
        f"- cloud required: `{str(summary.get('cloud_required')).lower()}`",
        "",
        "## Selector",
        "",
        f"- loss: `{summary.get('selector_loss')}`",
        f"- importance MSE: `{_fmt(summary.get('selector_test_importance_mse'))}`",
        f"- Pearson corr: `{_fmt(summary.get('selector_test_pearson_corr_mean'), 6)}`",
        f"- target topK overlap: `{_fmt(summary.get('selector_test_target_topk_overlap'), 6)}`",
        f"- selected teacher importance: `{_fmt(summary.get('selector_selected_teacher_importance_mean'), 6)}`",
        "",
        "## Baselines",
        "",
        "| policy | seed | student_future_mse | teacher_mse | ratio | topK overlap | selected importance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary.get("baseline_rows", []), key=lambda item: (str(item.get("policy")), int(item.get("seed", 0)))):
        policy = row.get("policy")
        if policy == "learned_selector":
            policy = "learned_selector_weighted_mse_alpha2"
        lines.append(
            "| {policy} | {seed} | {student} | {teacher} | {ratio} | {topk} | {importance} |".format(
                policy=policy,
                seed=row.get("seed"),
                student=_fmt(row.get("student_future_mse")),
                teacher=_fmt(row.get("teacher_mse")),
                ratio=_fmt(row.get("student_teacher_ratio"), 6),
                topk=_fmt(row.get("selector_target_topk_overlap"), 6),
                importance=_fmt(row.get("selected_teacher_importance_mean"), 6),
            )
        )
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            f"- learned vs random mean MSE delta: `{_fmt(summary.get('learned_vs_random_mse_delta'))}`",
            f"- learned vs uniform MSE delta: `{_fmt(summary.get('learned_vs_uniform_mse_delta'))}`",
            f"- learned vs teacher_importance_topk MSE delta: `{_fmt(summary.get('learned_vs_teacher_importance_topk_mse_delta'))}`",
            f"- learned selected importance vs random delta: `{_fmt(summary.get('learned_selected_importance_vs_random_delta'), 6)}`",
            f"- learned selected importance vs uniform delta: `{_fmt(summary.get('learned_selected_importance_vs_uniform_delta'), 6)}`",
            "",
            "## Caveats",
            "",
        ]
    )
    caveats = summary.get("caveats") or []
    lines.extend([f"- {item}" for item in caveats] or ["- none"])
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
            f"BAIR_500_64_SCALE_VALIDATION_PASS = {str(summary.get('sanity_gate_pass')).lower()}",
            "",
        ]
    )
    return "\n".join(lines)


def write_scale_validation_summary(
    config: dict[str, Any],
    run_root: str | Path | None = None,
    baseline_summary: str | Path | None = None,
    resource_summary: dict[str, Any] | None = None,
    pytest_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if run_root is not None:
        config = json.loads(json.dumps(config))
        config["output"]["summary_json"] = str(Path(run_root) / "scale_validation_summary.json")
        config["output"]["summary_md"] = str(Path(run_root) / "scale_validation_summary.md")
    payloads = load_step14_payloads(config, baseline_summary=baseline_summary)
    previous = _read_json(config["output"]["summary_json"], required=False)
    if resource_summary is None:
        resource_summary = previous.get("resource_summary", {})
    if pytest_result is None:
        pytest_result = previous.get("pytest_result")
    summary = build_scale_validation_summary_from_payloads(
        config=config,
        payloads=payloads,
        resource_summary=resource_summary,
        pytest_result=pytest_result,
    )
    json_path = Path(config["output"]["summary_json"])
    md_path = Path(config["output"]["summary_md"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(render_scale_validation_markdown(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-root")
    parser.add_argument("--baseline-summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_scale_validation_summary(
        load_yaml(args.config),
        run_root=args.run_root,
        baseline_summary=args.baseline_summary,
    )
    print("BAIR_500_64_SCALE_VALIDATION_SUMMARY_JSON")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
