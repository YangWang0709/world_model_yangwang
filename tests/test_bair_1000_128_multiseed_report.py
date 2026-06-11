"""Unit tests for Step 15 multi-seed summary generation with fake payloads."""

from __future__ import annotations

from eval.eval_bair_1000_128_multiseed_validation import (
    aggregate_rows_by_policy,
    build_multiseed_validation_summary_from_payloads,
    render_multiseed_validation_markdown,
    write_baseline_multiseed_aggregate,
    write_selector_multiseed_summary,
    write_student_multiseed_summary,
)


def _config() -> dict:
    return {
        "dataset": {"train_samples": 1000, "test_samples": 128, "subset_dir": "data/bair_robot_pushing_small_subset_1000_128"},
        "tokens": {"output_root": "data/token_shards/bair_videomae_1000_128"},
        "importance": {"output_root": "data/importance_shards/bair_videomae_teacher_1000_128"},
        "selector": {"topk": 16, "seeds": [0, 1, 2]},
    }


def _baseline_rows() -> list[dict]:
    return [
        {"policy": "random_k", "seed": 0, "student_future_mse": 1.8, "teacher_mse": 1.4, "student_teacher_ratio": 1.28, "selector_target_topk_overlap": 0.04, "selected_teacher_importance_mean": 0.50},
        {"policy": "random_k", "seed": 1, "student_future_mse": 1.9, "teacher_mse": 1.4, "student_teacher_ratio": 1.35, "selector_target_topk_overlap": 0.05, "selected_teacher_importance_mean": 0.52},
        {"policy": "random_k", "seed": 2, "student_future_mse": 2.0, "teacher_mse": 1.4, "student_teacher_ratio": 1.42, "selector_target_topk_overlap": 0.06, "selected_teacher_importance_mean": 0.51},
        {"policy": "uniform_k", "seed": 0, "student_future_mse": 1.7, "teacher_mse": 1.4, "student_teacher_ratio": 1.21, "selector_target_topk_overlap": 0.07, "selected_teacher_importance_mean": 0.53},
        {"policy": "teacher_importance_topk", "seed": 0, "student_future_mse": 1.45, "teacher_mse": 1.4, "student_teacher_ratio": 1.03, "selector_target_topk_overlap": 1.0, "selected_teacher_importance_mean": 0.90},
        {"policy": "learned_selector", "seed": 0, "student_future_mse": 1.60, "teacher_mse": 1.4, "student_teacher_ratio": 1.14, "selector_target_topk_overlap": 0.18, "selected_teacher_importance_mean": 0.61},
        {"policy": "learned_selector", "seed": 1, "student_future_mse": 1.62, "teacher_mse": 1.4, "student_teacher_ratio": 1.15, "selector_target_topk_overlap": 0.19, "selected_teacher_importance_mean": 0.62},
        {"policy": "learned_selector", "seed": 2, "student_future_mse": 1.64, "teacher_mse": 1.4, "student_teacher_ratio": 1.17, "selector_target_topk_overlap": 0.20, "selected_teacher_importance_mean": 0.63},
    ]


def _payloads(tmp_path) -> dict:
    selector_rows = [
        {"seed": 0, "success": True, "final_loss": 0.1, "test_importance_mse": 0.03, "test_importance_mae": 0.1, "test_pearson_corr_mean": 0.2, "test_target_topk_overlap": 0.18, "test_selected_teacher_importance_mean": 0.61, "test_selected_vs_random_importance_gap": 0.08},
        {"seed": 1, "success": True, "final_loss": 0.1, "test_importance_mse": 0.04, "test_importance_mae": 0.1, "test_pearson_corr_mean": 0.2, "test_target_topk_overlap": 0.19, "test_selected_teacher_importance_mean": 0.62, "test_selected_vs_random_importance_gap": 0.09},
        {"seed": 2, "success": True, "final_loss": 0.1, "test_importance_mse": 0.05, "test_importance_mae": 0.1, "test_pearson_corr_mean": 0.2, "test_target_topk_overlap": 0.20, "test_selected_teacher_importance_mean": 0.63, "test_selected_vs_random_importance_gap": 0.10},
    ]
    student_rows = [
        {"seed": 0, "success": True, "final_loss": 0.8, "student_future_mse": 1.60, "teacher_mse": 1.4, "student_teacher_ratio": 1.14, "selector_target_topk_overlap": 0.18, "selected_teacher_importance_mean": 0.61, "selected_vs_random_importance_gap": 0.08},
        {"seed": 1, "success": True, "final_loss": 0.8, "student_future_mse": 1.62, "teacher_mse": 1.4, "student_teacher_ratio": 1.15, "selector_target_topk_overlap": 0.19, "selected_teacher_importance_mean": 0.62, "selected_vs_random_importance_gap": 0.09},
        {"seed": 2, "success": True, "final_loss": 0.8, "student_future_mse": 1.64, "teacher_mse": 1.4, "student_teacher_ratio": 1.17, "selector_target_topk_overlap": 0.20, "selected_teacher_importance_mean": 0.63, "selected_vs_random_importance_gap": 0.10},
    ]
    selector_summary = {"rows": selector_rows, "aggregate": write_selector_multiseed_summary(selector_rows, tmp_path / "selector")["aggregate"]}
    student_summary = {"rows": student_rows, "aggregate": write_student_multiseed_summary(student_rows, tmp_path / "student")["aggregate"]}
    baseline_aggregate = {"aggregate": write_baseline_multiseed_aggregate(_baseline_rows(), tmp_path / "baseline")["aggregate"]}
    return {
        "subset": {"export_success": True, "splits": {"train": {"exported": 1000}, "test": {"exported": 128}}},
        "tokens": {"used_fallback": False, "split_summaries": {"train": {"output_token_shape": [4, 392, 768]}}},
        "teacher": {"initial_loss": 2.0, "final_loss": 1.0, "best_loss": 0.9, "run_dir": "runs/teacher"},
        "teacher_eval": {"eval_mse": 1.4},
        "importance": {"train_num_samples": 1000, "test_num_samples": 128},
        "importance_eval": {"num_samples": 1128, "importance_mean": 0.1, "importance_std": 0.2, "importance_min": -0.1, "importance_max": 1.0, "normalized_importance_mean": 0.5, "normalized_importance_std": 0.1, "normalized_importance_min": 0.0, "normalized_importance_max": 1.0, "base_loss_mean": 1.0, "masked_loss_mean": 1.1, "positive_importance_ratio": 0.6},
        "selector_multiseed": selector_summary,
        "student_multiseed": student_summary,
        "baseline": {"rows": _baseline_rows()},
        "baseline_aggregate": baseline_aggregate,
        "step14": {"train_samples": 500, "test_samples": 64, "baseline_learned_mse": 1.65, "baseline_random_mean_mse": 1.66, "baseline_uniform_mse": 1.68},
    }


def test_aggregate_rows_by_policy_supports_learned_multiseed() -> None:
    aggregate = aggregate_rows_by_policy(_baseline_rows())
    assert aggregate["baseline_learned_mse_mean"] == 1.62
    assert aggregate["baseline_random_mse_mean"] == 1.9
    assert aggregate["learned_beats_random_mse"] is True
    assert aggregate["learned_beats_uniform_mse"] is True
    assert aggregate["learned_selected_importance_beats_random"] is True
    assert aggregate["learned_vs_teacher_importance_topk_mse_delta"] > 0


def test_multiseed_summary_and_markdown(tmp_path) -> None:
    summary = build_multiseed_validation_summary_from_payloads(
        config=_config(),
        payloads=_payloads(tmp_path),
        resource_summary={"oom": False},
        pytest_result={"passed": True},
    )
    assert summary["sanity_gate_pass"] is True
    assert summary["token_retention_ratio"] == 16 / 392
    assert summary["baseline_learned_mse_mean"] == 1.62
    assert summary["baseline_random_mse_std"] > 0
    assert summary["selector_test_importance_mse_mean"] == 0.04
    markdown = render_multiseed_validation_markdown(summary)
    assert "learned_selector_weighted_mse_alpha2" in markdown
    assert "BAIR_1000_128_MULTI_SEED_VALIDATION_PASS = true" in markdown


def test_scientific_caveat_does_not_fail_engineering_gate(tmp_path) -> None:
    payloads = _payloads(tmp_path)
    for row in payloads["baseline"]["rows"]:
        if row["policy"] == "learned_selector":
            row["student_future_mse"] = 2.2
            row["selected_teacher_importance_mean"] = 0.30
    payloads["baseline_aggregate"] = {"aggregate": aggregate_rows_by_policy(payloads["baseline"]["rows"])}
    summary = build_multiseed_validation_summary_from_payloads(
        config=_config(),
        payloads=payloads,
        resource_summary={"oom": False},
        pytest_result={"passed": True},
    )
    assert summary["sanity_gate_pass"] is True
    assert summary["learned_beats_random_mse"] is False
    assert summary["scientific_caveats"]
