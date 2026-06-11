from eval.eval_bair_context_bottleneck_validation import build_context_bottleneck_summary, render_context_bottleneck_markdown


def _config(tmp_path):
    return {
        "dataset": {"output_dir": str(tmp_path / "windows"), "train_samples": 1000, "test_samples": 128, "context_len": 8, "current_len": 4, "future_len": 4},
        "tokens": {"output_root": str(tmp_path / "tokens")},
        "importance": {"output_root": str(tmp_path / "importance")},
        "teacher": {"run_name": "teacher", "checkpoint": str(tmp_path / "runs" / "teacher" / "ckpt.pt")},
        "unified_selector": {"run_name": "selector", "checkpoint": str(tmp_path / "runs" / "selector" / "ckpt.pt")},
        "context_bottleneck_world_model": {"run_name": "world", "checkpoint": str(tmp_path / "runs" / "world" / "ckpt.pt")},
        "baselines": {"run_name": "baseline"},
        "output": {"run_root": str(tmp_path / "runs"), "run_name": "main", "summary_json": str(tmp_path / "runs" / "main" / "summary.json"), "summary_md": str(tmp_path / "runs" / "main" / "summary.md")},
    }


def test_context_bottleneck_summary_comparison_and_caveat(tmp_path):
    cfg = _config(tmp_path)
    (tmp_path / "runs" / "baseline").mkdir(parents=True)
    (tmp_path / "runs" / "baseline" / "baseline_aggregate.json").write_text(
        '{"aggregate":[{"policy":"current_only","student_future_mse_mean":1.0},{"policy":"random_context_topK","student_future_mse_mean":1.2},{"policy":"uniform_context_topK","student_future_mse_mean":1.1},{"policy":"teacher_context_importance_topK","student_future_mse_mean":0.9},{"policy":"learned_context_selector_topK","student_future_mse_mean":1.3},{"policy":"hybrid_context_learned_uniform","student_future_mse_mean":1.25}]}',
        encoding="utf-8",
    )
    summary = build_context_bottleneck_summary(config=cfg, resource_summary={"oom": False}, pytest_result={"passed": True}, env_guard={"tensorflow": False, "tensorflow_datasets": False})
    assert summary["learned_beats_current_only"] is False
    assert "BAIR short-window" in " ".join(summary["scientific_caveats"])
    assert "BAIR_CONTEXT_BOTTLENECK_VALIDATION_PASS" in render_context_bottleneck_markdown(summary)
