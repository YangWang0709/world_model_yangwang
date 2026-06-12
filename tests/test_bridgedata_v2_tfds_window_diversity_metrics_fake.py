from data.bridgedata_v2_tfds_window_diversity_metrics import (
    aggregate_window_diversity_metrics,
    recommended_step30b_from_signal,
)


def _metric(policy, loss):
    return {"policy": policy, "val_final_loss": loss, "val_best_loss": loss}


def test_step30a_window_diversity_metrics_aggregate_policy_means_and_counts():
    runs = {
        "runs": [
            {
                "context_comparison": {
                    "proxy_better_than_current_only": True,
                    "proxy_better_than_random": True,
                    "full_better_than_current_only": False,
                },
                "policy_metrics": [
                    _metric("current_only", 4.0),
                    _metric("random_context_topk", 4.2),
                    _metric("proxy_importance_topk", 3.8),
                    _metric("full_context_reference", 4.4),
                ],
            },
            {
                "context_comparison": {
                    "proxy_better_than_current_only": False,
                    "proxy_better_than_random": True,
                    "full_better_than_current_only": True,
                },
                "policy_metrics": [
                    _metric("current_only", 5.0),
                    _metric("random_context_topk", 5.2),
                    _metric("proxy_importance_topk", 5.1),
                    _metric("full_context_reference", 4.7),
                ],
            },
        ]
    }
    summary = aggregate_window_diversity_metrics(runs)
    assert summary["num_runs"] == 2
    assert summary["policy_val_final_mean"]["current_only"] == 4.5
    assert summary["proxy_beats_current_count"] == 1
    assert summary["proxy_beats_random_count"] == 2
    assert summary["full_beats_current_count"] == 1
    assert summary["all_val_losses_finite"] is True


def test_step30a_recommended_next_step_is_available_for_all_signals():
    for signal in [
        "positive_but_not_final",
        "weak_proxy_positive_full_context_noisy",
        "inconclusive",
        "negative_or_current_dominant",
    ]:
        rec = recommended_step30b_from_signal(signal)
        assert rec["name"]
        assert "selector" in rec["scope"] or signal == "negative_or_current_dominant"
