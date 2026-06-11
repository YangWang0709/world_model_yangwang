"""Unit tests for Step 14 summary generation with fake payloads."""

from __future__ import annotations

from eval.eval_bair_500_64_scale_validation import (
    build_scale_validation_summary_from_payloads,
    compute_baseline_comparisons,
    render_scale_validation_markdown,
)


def _config() -> dict:
    return {
        "dataset": {"train_samples": 500, "test_samples": 64, "subset_dir": "data/bair_robot_pushing_small_subset_500_64"},
        "tokens": {"output_root": "data/token_shards/bair_videomae_500_64"},
        "importance": {"output_root": "data/importance_shards/bair_videomae_teacher_500_64"},
        "selector": {"topk": 16},
    }


def _payloads() -> dict:
    rows = [
        {"policy": "random_k", "seed": 0, "student_future_mse": 2.0, "teacher_mse": 1.5, "student_teacher_ratio": 1.33, "selector_target_topk_overlap": 0.04, "selected_teacher_importance_mean": 0.40},
        {"policy": "random_k", "seed": 1, "student_future_mse": 1.8, "teacher_mse": 1.5, "student_teacher_ratio": 1.20, "selector_target_topk_overlap": 0.05, "selected_teacher_importance_mean": 0.42},
        {"policy": "uniform_k", "seed": 0, "student_future_mse": 1.7, "teacher_mse": 1.5, "student_teacher_ratio": 1.13, "selector_target_topk_overlap": 0.06, "selected_teacher_importance_mean": 0.45},
        {"policy": "teacher_importance_topk", "seed": 0, "student_future_mse": 1.55, "teacher_mse": 1.5, "student_teacher_ratio": 1.03, "selector_target_topk_overlap": 1.0, "selected_teacher_importance_mean": 0.90},
        {"policy": "learned_selector", "seed": 0, "student_future_mse": 1.6, "teacher_mse": 1.5, "student_teacher_ratio": 1.07, "selector_target_topk_overlap": 0.20, "selected_teacher_importance_mean": 0.60},
    ]
    return {
        "subset": {"export_success": True, "splits": {"train": {"exported": 500}, "test": {"exported": 64}}},
        "tokens": {"used_fallback": False, "split_summaries": {"train": {"output_token_shape": [4, 392, 768]}}},
        "teacher": {"initial_loss": 2.0, "final_loss": 1.0, "best_loss": 0.9, "run_dir": "runs/teacher"},
        "teacher_eval": {"eval_mse": 1.5},
        "importance": {"train_num_samples": 500, "test_num_samples": 64},
        "importance_eval": {"num_samples": 564, "importance_mean": 0.1, "importance_std": 0.2, "importance_min": -0.1, "importance_max": 1.0, "normalized_importance_mean": 0.5, "normalized_importance_std": 0.1, "normalized_importance_min": 0.0, "normalized_importance_max": 1.0, "base_loss_mean": 1.0, "masked_loss_mean": 1.1, "positive_importance_ratio": 0.6},
        "selector": {"test_importance_mse": 0.03, "test_pearson_corr_mean": 0.1, "test_target_topk_overlap": 0.2, "test_selected_teacher_importance_mean": 0.6, "num_tokens": 392},
        "student": {"student_future_mse": 1.6, "run_dir": "runs/student"},
        "student_eval": {"student_future_mse": 1.6},
        "gap": {"student_future_mse": 1.6, "teacher_mse": 1.5, "student_teacher_ratio": 1.0666},
        "baseline": {"rows": rows},
    }


def test_compute_baseline_comparisons() -> None:
    comparison = compute_baseline_comparisons(_payloads()["baseline"]["rows"])
    assert comparison["baseline_random_mean_mse"] == 1.9
    assert comparison["learned_beats_random_mse"] is True
    assert comparison["learned_beats_uniform_mse"] is True
    assert comparison["learned_selected_importance_beats_random"] is True
    assert comparison["learned_vs_teacher_importance_topk_mse_delta"] > 0


def test_scale_validation_summary_and_markdown() -> None:
    summary = build_scale_validation_summary_from_payloads(
        config=_config(),
        payloads=_payloads(),
        resource_summary={"oom": False},
        pytest_result={"passed": True},
    )
    assert summary["sanity_gate_pass"] is True
    assert summary["token_retention_ratio"] == 16 / 392
    assert summary["selector_loss"] == "weighted_mse_alpha2"
    markdown = render_scale_validation_markdown(summary)
    assert "learned_selector_weighted_mse_alpha2" in markdown
    assert "BAIR_500_64_SCALE_VALIDATION_PASS = true" in markdown


def test_summary_records_caveat_without_failing_engineering_gate() -> None:
    payloads = _payloads()
    for row in payloads["baseline"]["rows"]:
        if row["policy"] == "learned_selector":
            row["student_future_mse"] = 2.2
            row["selected_teacher_importance_mean"] = 0.30
    summary = build_scale_validation_summary_from_payloads(
        config=_config(),
        payloads=payloads,
        resource_summary={"oom": False},
        pytest_result={"passed": True},
    )
    assert summary["sanity_gate_pass"] is True
    assert summary["learned_beats_random_mse"] is False
    assert summary["caveats"]
