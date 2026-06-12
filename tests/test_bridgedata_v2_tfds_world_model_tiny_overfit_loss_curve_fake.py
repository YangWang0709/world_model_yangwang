from data.bridgedata_v2_tfds_world_model_training_metrics import build_policy_comparison, summarize_loss_curve


def test_loss_curve_summary_schema_and_decrease():
    curve = [
        {"step": 0, "loss": 10.0},
        {"step": 5, "loss": 8.0},
        {"step": 10, "loss": 2.5},
    ]
    summary = summarize_loss_curve("proxy_importance_topk", curve, min_relative_loss_decrease=0.01)
    assert summary["policy"] == "proxy_importance_topk"
    assert summary["initial_loss"] == 10.0
    assert summary["final_loss"] == 2.5
    assert summary["best_loss"] == 2.5
    assert summary["absolute_loss_decrease"] == 7.5
    assert summary["relative_loss_decrease"] == 0.75
    assert summary["loss_decreased"] is True
    assert summary["all_losses_finite"] is True


def test_policy_comparison_acceptance_counts_loss_decreases():
    results = [
        {
            "policy": "current_only",
            "topk": 0,
            "initial_loss": 10.0,
            "final_loss": 5.0,
            "best_loss": 5.0,
            "absolute_loss_decrease": 5.0,
            "relative_loss_decrease": 0.5,
            "all_losses_finite": True,
        },
        {
            "policy": "random_context_topk",
            "topk": 256,
            "initial_loss": 10.0,
            "final_loss": 9.95,
            "best_loss": 9.95,
            "absolute_loss_decrease": 0.05,
            "relative_loss_decrease": 0.005,
            "all_losses_finite": True,
        },
    ]
    comparison = build_policy_comparison(results, min_relative_loss_decrease=0.01)
    assert comparison["num_policies"] == 2
    assert comparison["policies_with_loss_decrease"] == 1
    assert comparison["all_losses_finite"] is True
    assert comparison["best_policy_by_final_loss"] == "current_only"
    assert comparison["loss_quality_note"] == "tiny-overfit only; not final performance"
