import json
from pathlib import Path

from eval.eval_bair_context_selector_oracle_gap import write_context_selector_oracle_gap_summary


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_context_selector_oracle_gap_report_fake_outputs(tmp_path: Path):
    run_dir = tmp_path / "context_selector_oracle_gap_bair_1000_128_v1"
    _write(
        run_dir / "label_diagnostic" / "context_importance_diagnostic.json",
        {
            "importance_stats": {"mean": 0.0, "std": 0.1, "min": -1.0, "max": 1.0},
            "importance_norm_stats": {"mean": 0.5, "std": 0.1, "min": 0.0, "max": 1.0},
            "positive_importance_ratio": 0.5,
            "topk_concentration": {"top32_mass_ratio": 0.4},
            "temporal_block_stats": {"oracle_topk_block_distribution": [4.0] * 8},
            "oracle_random_gap": {"oracle_vs_random_importance_gap": 0.2},
            "label_sparse_or_noisy": False,
        },
    )
    selector_rows = [
        {"variant": "weighted_mse_alpha2", "target_topk_overlap": 0.2, "selected_context_importance_mean": 0.6},
        {"variant": "weighted_mse_alpha2_no_current_condition", "target_topk_overlap": 0.1, "selected_context_importance_mean": 0.5},
        {"variant": "weighted_mse_alpha2_no_temporal_pos", "target_topk_overlap": 0.15, "selected_context_importance_mean": 0.55},
        {"variant": "hybrid_weighted_mse_rank_bce", "target_topk_overlap": 0.25, "selected_context_importance_mean": 0.65},
        {"variant": "temporal_block_balanced_topk", "target_topk_overlap": 0.3, "selected_context_importance_mean": 0.66},
    ]
    _write(run_dir / "selector_phase_a_summary.json", {"rows": selector_rows * 2})
    _write(run_dir / "downstream_phase_a_summary.json", {"rows": [{"variant": "temporal_block_balanced_topk", "future_mse": 1.30}]})
    _write(
        run_dir / "selector_phase_b_aggregate.json",
        {"aggregate": [{"variant": "temporal_block_balanced_topk", "seeds": "0,1,2", "n": 3, "topk_overlap_mean": 0.3, "selected_importance_mean": 0.66}]},
    )
    _write(
        run_dir / "downstream_phase_b_aggregate.json",
        {"aggregate": [{"variant": "temporal_block_balanced_topk", "seeds": "0,1,2", "n": 3, "future_mse_mean": 1.30, "future_mse_std": 0.01, "oracle_gap_mean": 0.08}]},
    )
    _write(run_dir / "runner_summary.json", {"current_tokens_dropped": False, "trained_current_importance": False, "trained_context_importance": True, "resource_summary": {"oom": False}})
    step17_summary = tmp_path / "step17.json"
    step17_baseline = tmp_path / "baseline.json"
    _write(
        step17_summary,
        {
            "current_only_mse": 1.39,
            "random_context_mse": 1.36,
            "learned_context_mse": 1.37,
            "teacher_context_importance_topk_mse": 1.22,
        },
    )
    _write(step17_baseline, {"aggregate": []})
    summary = write_context_selector_oracle_gap_summary(
        run_dir=run_dir,
        step17_summary_path=step17_summary,
        step17_baseline_path=step17_baseline,
        pytest_result={"passed": True},
        env_guard={"tensorflow": False, "tensorflow_datasets": False},
    )
    assert summary["phase_b_best_variant"] == "temporal_block_balanced_topk"
    assert summary["oracle_gap_reduction_vs_step17"] > 0
    assert summary["best_beats_random_context"] is True
    assert (run_dir / "context_selector_oracle_gap_summary.md").exists()

