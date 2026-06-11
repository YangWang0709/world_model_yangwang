from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.selector_ablation_trainer import (
    aggregate_downstream_rows,
    aggregate_selector_rows,
    render_downstream_markdown,
    render_selector_markdown,
    write_downstream_ablation_summaries,
    write_selector_ablation_summaries,
)


def _selector_row(name: str, loss_type: str, mse: float, overlap: float, importance: float) -> dict:
    return {
        "variant_name": name,
        "variant_run_name": f"{name}_seed0",
        "loss_type": loss_type,
        "seed": 0,
        "success": True,
        "final_loss": mse,
        "test_importance_mse": mse,
        "test_importance_mae": mse,
        "test_pearson_corr_mean": 0.5,
        "test_target_top1_overlap": 0.25,
        "test_target_topk_overlap": overlap,
        "test_selected_teacher_importance_mean": importance,
        "test_random_teacher_importance_mean": 0.4,
        "test_selected_vs_random_importance_gap": importance - 0.4,
    }


def _downstream_row(name: str, mse: float, importance: float, overlap: float) -> dict:
    return {
        "variant_name": name,
        "variant_run_name": f"{name}_seed0",
        "loss_type": name,
        "seed": 0,
        "success": True,
        "final_loss": mse,
        "student_future_mse": mse,
        "teacher_mse": 0.1,
        "student_teacher_ratio": mse / 0.1,
        "token_retention_ratio": 16 / 392,
        "selector_target_top1_overlap": 0.25,
        "selector_target_topk_overlap": overlap,
        "selected_teacher_importance_mean": importance,
        "random_teacher_importance_mean": 0.4,
        "selected_vs_random_importance_gap": importance - 0.4,
    }


def _write_step12(path: Path) -> None:
    payload = {
        "rows": [
            _downstream_row("random_k", 0.7, 0.42, 0.04) | {"policy": "random_k"},
            _downstream_row("random_k", 0.6, 0.44, 0.05) | {"policy": "random_k"},
            _downstream_row("uniform_k", 0.55, 0.48, 0.06) | {"policy": "uniform_k"},
            _downstream_row("teacher_importance_topk", 0.3, 0.90, 1.0)
            | {"policy": "teacher_importance_topk"},
            _downstream_row("learned_selector", 0.58, 0.60, 0.16) | {"policy": "learned_selector"},
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_selector_ablation_aggregates_pick_best_variants() -> None:
    rows = [
        _selector_row("mse_only", "mse_only", 0.3, 0.10, 0.50),
        _selector_row("hybrid", "hybrid_weighted_mse_rank_bce", 0.2, 0.25, 0.70),
    ]

    aggregate = aggregate_selector_rows(rows)
    markdown = render_selector_markdown(rows, aggregate)

    assert aggregate["sanity_gate"]["pass"] is True
    assert aggregate["best_by_selected_teacher_importance"]["variant_name"] == "hybrid"
    assert aggregate["best_by_importance_mse"]["variant_name"] == "hybrid"
    assert "selected_teacher_importance_mean" in markdown


def test_downstream_ablation_compares_against_step12(tmp_path: Path) -> None:
    step12 = tmp_path / "step12.json"
    _write_step12(step12)
    rows = [
        _downstream_row("mse_only", 0.62, 0.55, 0.12),
        _downstream_row("hybrid", 0.50, 0.75, 0.30),
    ]

    aggregate = aggregate_downstream_rows(rows, step12_summary_json=step12)
    markdown = render_downstream_markdown(rows, aggregate)

    assert aggregate["sanity_gate"]["pass"] is True
    assert aggregate["best_by_student_future_mse"]["variant_name"] == "hybrid"
    assert aggregate["comparison"]["downstream_improvement_over_step12_learned"] is True
    assert aggregate["comparison"]["any_variant_selected_importance_gt_step12_learned"] is True
    assert "student_future_mse" in markdown


def test_ablation_summary_writers_handle_nonfinite_without_crashing(tmp_path: Path) -> None:
    selector_rows = [
        _selector_row("mse_only", "mse_only", 0.3, 0.1, 0.5),
        _selector_row("bad", "topk_bce", float("inf"), float("nan"), 0.0),
    ]
    downstream_rows = [_downstream_row("mse_only", 0.6, 0.5, 0.1)]
    step12 = tmp_path / "step12.json"
    _write_step12(step12)

    selector_summary = write_selector_ablation_summaries(selector_rows, tmp_path)
    downstream_summary = write_downstream_ablation_summaries(
        downstream_rows,
        tmp_path,
        step12_summary_json=step12,
        selector_summary={"rows": selector_rows, "aggregate": selector_summary["selector_aggregate"]},
    )

    assert Path(selector_summary["selector_summary_json"]).exists()
    assert Path(downstream_summary["combined_report_json"]).exists()
    assert selector_summary["selector_aggregate"]["sanity_gate"]["pass"] is False
    assert downstream_summary["downstream_aggregate"]["comparison"][
        "best_downstream_vs_step12_learned_mse_delta"
    ] == pytest.approx(0.02)
