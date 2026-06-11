from eval.eval_bair_downstream_utilization_ablation import (
    build_downstream_summary,
    render_downstream_markdown,
)


def _config():
    return {
        "data": {
            "max_train_samples": 1000,
            "max_test_samples": 128,
            "train_token_shard_dir": "/x/bair_videomae_1000_128/train",
            "test_token_shard_dir": "/x/bair_videomae_1000_128/test",
            "train_importance_shard_dir": "/x/bair_videomae_teacher_1000_128/train",
            "test_importance_shard_dir": "/x/bair_videomae_teacher_1000_128/test",
        },
        "selection": {"topk": 16, "token_retention_ratio": 16 / 392},
        "teacher_reference": {"checkpoint": "/x/teacher.pt"},
        "learned_selectors": {"root": "/x/selectors"},
        "step15_reference": {
            "baseline_aggregate": "/x/baseline.json",
            "learned_mse_mean": 1.57323252,
            "random_mse_mean": 1.57845018,
            "uniform_mse": 1.56802237,
            "teacher_importance_topk_mse": 1.52461295,
            "learned_selected_importance_mean": 0.54776923,
        },
    }


def test_downstream_summary_selects_best_and_compares_step15():
    phase_a_rows = [
        {"phase": "phase_a", "variant": "a", "seed": 0, "success": True, "student_future_mse": 1.6, "teacher_mse": 1.2, "student_teacher_ratio": 1.33},
        {"phase": "phase_a", "variant": "b", "seed": 0, "success": True, "student_future_mse": 1.5, "teacher_mse": 1.2, "student_teacher_ratio": 1.25},
    ]
    phase_b_rows = [
        {"phase": "phase_b", "variant": "b", "seed": 0, "success": True, "student_future_mse": 1.50, "teacher_mse": 1.2, "student_teacher_ratio": 1.25, "selected_teacher_importance_mean": 0.56, "selector_target_topk_overlap": 0.12},
        {"phase": "phase_b", "variant": "b", "seed": 1, "success": True, "student_future_mse": 1.52, "teacher_mse": 1.2, "student_teacher_ratio": 1.27, "selected_teacher_importance_mean": 0.57, "selector_target_topk_overlap": 0.13},
        {"phase": "phase_b", "variant": "b", "seed": 2, "success": True, "student_future_mse": 1.51, "teacher_mse": 1.2, "student_teacher_ratio": 1.26, "selected_teacher_importance_mean": 0.58, "selector_target_topk_overlap": 0.14},
        {"phase": "phase_b", "variant": "hybrid_learned8_uniform8_cross_attention", "seed": 0, "success": True, "student_future_mse": 1.56, "teacher_mse": 1.2, "student_teacher_ratio": 1.30, "selected_teacher_importance_mean": 0.55, "selector_target_topk_overlap": 0.11},
        {"phase": "phase_b", "variant": "hybrid_learned8_uniform8_cross_attention", "seed": 1, "success": True, "student_future_mse": 1.55, "teacher_mse": 1.2, "student_teacher_ratio": 1.29, "selected_teacher_importance_mean": 0.55, "selector_target_topk_overlap": 0.11},
        {"phase": "phase_b", "variant": "hybrid_learned8_uniform8_cross_attention", "seed": 2, "success": True, "student_future_mse": 1.54, "teacher_mse": 1.2, "student_teacher_ratio": 1.28, "selected_teacher_importance_mean": 0.55, "selector_target_topk_overlap": 0.11},
    ]
    summary = build_downstream_summary(
        config=_config(),
        run_dir="/tmp/run",
        phase_a_rows=phase_a_rows,
        phase_b_rows=phase_b_rows,
        resource_summary={"oom": False},
        pytest_result={"passed": True},
        env_guard={"tensorflow": False, "tensorflow_datasets": False},
    )
    assert summary["phase_a_best_variant"] == "b"
    assert summary["phase_b_best_mse_variant"] == "b"
    assert summary["best_beats_step15_learned"] is True
    assert summary["best_beats_step15_uniform"] is True
    assert summary["best_beats_step15_random_mean"] is True
    assert summary["sanity_gate_pass"] is False
    assert "BAIR_DOWNSTREAM_UTILIZATION_ABLATION_PASS" in render_downstream_markdown(summary)
