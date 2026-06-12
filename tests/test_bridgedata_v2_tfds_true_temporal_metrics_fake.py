import pytest

from data.bridgedata_v2_tfds_true_temporal_metrics import summarize_true_temporal_trainval


def test_true_temporal_metrics_compare_against_frame_repeat_and_recommend_next_step():
    step32 = {
        "horizon_target_rows": [
            {
                "horizon_gap": 0,
                "target_variant": "future_delta_last_minus_current",
                "mean_proxy_gain_over_current": 0.20,
                "proxy_beats_current_fraction": 1.0,
                "proxy_beats_random_fraction": 1.0,
                "full_context_noise_penalty_present": True,
            }
        ]
    }
    runs = [
        {
            "target_variant": "future_delta_last_minus_current",
            "split_seed": seed,
            "policy_val": {
                "current_only": 10.0,
                "random_context_topk": 8.0,
                "proxy_importance_topk": 6.0,
                "full_context_reference": 7.0,
            },
        }
        for seed in [42, 123, 999]
    ]
    summary = summarize_true_temporal_trainval(
        runs,
        step32_horizon_target_summary=step32,
        positive_thresholds={
            "true_temporal_proxy_beats_current_fraction_min": 0.67,
            "true_temporal_proxy_beats_random_fraction_min": 0.67,
            "true_temporal_gain_over_frame_repeat_min": 0.05,
        },
    )
    assert summary["frame_repeat_proxy_gain_mean"] == 0.20
    assert summary["true_temporal_proxy_gain_mean"] == pytest.approx(0.40)
    assert summary["true_temporal_gain_over_frame_repeat"] > 0.05
    assert summary["true_temporal_representation_helped"] is True
    assert summary["context_utility_claim_allowed"] is False
    assert summary["selector_training_allowed"] is False
    assert summary["current_importance_training_allowed"] is False
