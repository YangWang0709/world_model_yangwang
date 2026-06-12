from data.bridgedata_v2_tfds_context_signal_stability import analyze_context_signal_stability


def _run(seed, proxy_current, proxy_random, full_current):
    return {
        "split_seed": seed,
        "context_comparison": {
            "proxy_better_than_current_only": proxy_current,
            "proxy_better_than_random": proxy_random,
            "full_better_than_current_only": full_current,
            "proxy_val_relative_improvement_over_current_only": 0.02 if proxy_current else -0.01,
            "proxy_val_relative_improvement_over_random": 0.02 if proxy_random else -0.01,
            "full_val_relative_improvement_over_current_only": 0.02 if full_current else -0.01,
        },
        "policy_metrics": [],
    }


def test_step30a_stability_positive_but_not_final():
    summary = analyze_context_signal_stability({"runs": [_run(1, True, True, True), _run(2, True, True, True), _run(3, True, True, False)]})
    assert summary["context_signal_stability"] == "positive_but_not_final"
    assert summary["context_utility_claim_allowed"] is False
    assert summary["selector_training_allowed"] is False
    assert summary["current_importance_training_allowed"] is False


def test_step30a_stability_weak_proxy_positive_full_noisy():
    summary = analyze_context_signal_stability({"runs": [_run(1, True, True, False), _run(2, True, True, False), _run(3, False, True, True)]})
    assert summary["context_signal_stability"] == "weak_proxy_positive_full_context_noisy"
    assert summary["context_utility_claim_allowed"] is False


def test_step30a_stability_inconclusive_and_negative():
    inconclusive = analyze_context_signal_stability({"runs": [_run(1, True, True, False), _run(2, False, True, False), _run(3, False, False, False)]})
    negative = analyze_context_signal_stability({"runs": [_run(1, False, False, False), _run(2, False, True, False), _run(3, True, False, False)]})
    assert inconclusive["context_signal_stability"] == "inconclusive"
    assert negative["context_signal_stability"] == "negative_or_current_dominant"
