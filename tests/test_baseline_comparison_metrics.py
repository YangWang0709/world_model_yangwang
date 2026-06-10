from __future__ import annotations

import pytest

from training.baseline_student_world_model_trainer import aggregate_baseline_rows, render_baseline_markdown


def _row(policy: str, seed: int, mse: float, coverage: float, ratio: float = 1.0) -> dict:
    return {
        "policy": policy,
        "seed": seed,
        "final_loss": mse,
        "student_future_mse": mse,
        "teacher_future_mse": 0.1,
        "student_teacher_gap": mse - 0.1,
        "student_teacher_ratio": ratio,
        "token_retention_ratio": 0.25,
        "selected_top1_hit_rate": coverage,
        "selected_topk_hit_rate": coverage,
        "selected_key_coverage": coverage,
    }


def test_aggregate_baseline_rows_computes_random_mean_and_sanity_gate() -> None:
    rows = [
        _row("random_k", 0, 0.4, 0.1),
        _row("random_k", 1, 0.2, 0.3),
        _row("oracle_key", 0, 0.1, 1.0),
        _row("learned_selector", 0, 0.15, 1.0, ratio=1.5),
    ]

    aggregate = aggregate_baseline_rows(rows)

    assert aggregate["random_k_student_future_mse"]["mean"] == pytest.approx(0.3)
    assert aggregate["random_k_selected_key_coverage"]["mean"] == pytest.approx(0.2)
    assert aggregate["learned_vs_random"]["future_mse_lower_than_random_mean"] is True
    assert aggregate["learned_vs_random"]["coverage_higher_than_random_mean"] is True
    assert aggregate["sanity_gate"]["pass"] is True


def test_render_baseline_markdown_contains_expected_columns() -> None:
    rows = [
        _row("random_k", 0, 0.4, 0.1),
        _row("learned_selector", 0, 0.15, 1.0),
    ]
    aggregate = aggregate_baseline_rows(rows)

    markdown = render_baseline_markdown(rows, aggregate)

    assert "| policy | seed | final_loss |" in markdown
    assert "learned_selector" in markdown
    assert "random_k" in markdown
