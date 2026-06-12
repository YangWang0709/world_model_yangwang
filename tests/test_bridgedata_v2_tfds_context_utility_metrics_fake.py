from data.bridgedata_v2_tfds_context_utility_metrics import build_context_utility_comparison


def _metrics(current, random, proxy, full):
    return [
        {"policy": "current_only", "val_final_loss": current},
        {"policy": "random_context_topk", "val_final_loss": random},
        {"policy": "proxy_importance_topk", "val_final_loss": proxy},
        {"policy": "full_context_reference", "val_final_loss": full},
    ]


def test_context_utility_positive_signal_still_forbids_final_claim():
    result = build_context_utility_comparison(_metrics(10.0, 9.0, 8.0, 7.0))
    assert result["context_utility_sanity_signal"] == "positive"
    assert result["context_utility_claim_allowed"] is False
    assert result["proxy_better_than_current_only"] is True
    assert result["train_current_importance"] is False


def test_context_utility_inconclusive_when_full_helps_but_proxy_does_not():
    result = build_context_utility_comparison(_metrics(10.0, 9.0, 9.5, 8.0))
    assert result["context_utility_sanity_signal"] == "inconclusive_teacher_or_selector_issue"
    assert result["context_utility_claim_allowed"] is False


def test_context_utility_negative_when_current_only_best():
    result = build_context_utility_comparison(_metrics(8.0, 9.0, 10.0, 11.0))
    assert result["context_utility_sanity_signal"] == "negative_or_current_dominant"
    assert result["context_utility_claim_allowed"] is False

