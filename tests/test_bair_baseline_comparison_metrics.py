from __future__ import annotations

import pytest

from training.baseline_student_world_model_trainer import aggregate_baseline_rows, render_baseline_markdown


def _row(
    policy: str,
    seed: int,
    mse: float,
    importance: float,
    ratio: float = 1.0,
) -> dict:
    return {
        "policy": policy,
        "seed": seed,
        "final_loss": mse,
        "student_future_mse": mse,
        "teacher_mse": 0.1,
        "teacher_future_mse": 0.1,
        "student_teacher_gap": mse - 0.1,
        "student_teacher_ratio": ratio,
        "token_retention_ratio": 0.040816,
        "selector_target_top1_overlap": 0.0,
        "selector_target_topk_overlap": 0.25,
        "selected_teacher_importance_mean": importance,
        "random_teacher_importance_mean": 0.4,
        "selected_vs_random_importance_gap": importance - 0.4,
    }


def test_bair_baseline_aggregate_sanity_gate_uses_importance() -> None:
    rows = [
        _row("random_k", 0, 0.5, 0.40),
        _row("random_k", 1, 0.7, 0.44),
        _row("teacher_importance_topk", 0, 0.3, 0.90),
        _row("learned_selector", 0, 0.6, 0.60, ratio=1.5),
    ]

    aggregate = aggregate_baseline_rows(rows)

    assert aggregate["random_k_student_future_mse"]["mean"] == pytest.approx(0.6)
    assert aggregate["random_k_selected_teacher_importance_mean"]["mean"] == pytest.approx(0.42)
    assert aggregate["learned_vs_random"]["selected_teacher_importance_higher_than_random_mean"] is True
    assert aggregate["learned_vs_random"]["future_mse_lower_than_random_mean"] is True
    assert aggregate["learned_vs_oracle"]["selected_teacher_importance_gap"] == pytest.approx(-0.30)
    assert aggregate["sanity_gate"]["pass"] is True


def test_bair_baseline_markdown_table_renders_importance_columns() -> None:
    rows = [
        _row("random_k", 0, 0.5, 0.40),
        _row("learned_selector", 0, 0.4, 0.60),
        _row("teacher_importance_topk", 0, 0.3, 0.90),
    ]
    aggregate = aggregate_baseline_rows(rows)

    markdown = render_baseline_markdown(rows, aggregate)

    assert "| policy | seed | final_loss |" in markdown
    assert "selected_teacher_importance_mean" in markdown
    assert "teacher_importance_topk" in markdown
