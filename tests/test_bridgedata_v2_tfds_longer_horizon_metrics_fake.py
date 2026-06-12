import torch

from data.bridgedata_v2_tfds_longer_horizon_metrics import (
    recommend_step33,
    summarize_longer_horizon_trainval,
    target_summary_for_variant,
)


def test_target_variants_include_delta_targets():
    sample = {
        "current_tokens": torch.ones(4, 392, 3),
        "future_tokens": torch.full((4, 392, 3), 3.0),
    }
    assert target_summary_for_variant(sample, "future_mean_all4").shape == (3,)
    assert torch.allclose(target_summary_for_variant(sample, "future_delta_last_minus_current"), torch.full((3,), 2.0))
    assert torch.allclose(target_summary_for_variant(sample, "future_delta_mean_minus_current"), torch.full((3,), 2.0))


def test_longer_horizon_metrics_aggregate_delta_gain_and_step33():
    runs = []
    for gap, delta_proxy in [(0, 1.6), (4, 1.4), (8, 1.2)]:
        runs.append(_run(gap, "future_mean_all4", 42, current=3.0, random=2.8, proxy=2.7, full=3.2))
        runs.append(_run(gap, "future_delta_last_minus_current", 42, current=3.0, random=2.6, proxy=delta_proxy, full=2.9))
    summary = summarize_longer_horizon_trainval(runs, delta_gain_threshold=0.10)
    assert summary["delta_target_amplifies_context_gain"] is True
    assert summary["proxy_beats_current_fraction"] == 1.0
    assert summary["proxy_beats_random_fraction"] == 1.0
    assert summary["context_utility_claim_allowed"] is False
    assert summary["selector_training_allowed"] is False
    assert summary["recommended_step33"]["scope"] == "no selector/current-importance training yet"


def test_recommend_step33_never_allows_selector_scope():
    rec = recommend_step33(
        delta_amplifies=True,
        longer_horizon_increases=True,
        full_context_noise_confirmed=True,
        best_horizon_gap=8,
        best_target_variant="future_delta_last_minus_current",
    )
    assert "true temporal" in rec["name"]
    assert rec["scope"] == "no selector/current-importance training yet"


def _run(gap: int, target: str, seed: int, *, current: float, random: float, proxy: float, full: float) -> dict:
    return {
        "horizon_gap": gap,
        "target_variant": target,
        "split_seed": seed,
        "policy_val": {
            "current_only": current,
            "random_context_topk": random,
            "proxy_importance_topk": proxy,
            "full_context_reference": full,
        },
    }

