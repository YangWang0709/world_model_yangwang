from data.bridgedata_v2_tfds_cross_shard_metrics import (
    aggregate_eval_rows,
    decide_cross_shard_stability,
    skipped_eval_summary,
)


def _row(seed: int, proxy: float = 0.5):
    return {
        "split_seed": seed,
        "policy_val": {
            "current_only": 1.0,
            "random_context_topk": 0.8,
            "proxy_importance_topk": proxy,
            "full_context_reference": 1.2,
        },
    }


def test_cross_shard_metrics_aggregate_and_decide_stable():
    within = aggregate_eval_rows([_row(42), _row(123), _row(999)], eval_type="within_shard")
    cross = aggregate_eval_rows([_row(42), _row(123), _row(999)], eval_type="cross_shard")
    mixed = aggregate_eval_rows([_row(42), _row(123), _row(999)], eval_type="mixed_shard")
    decision = decide_cross_shard_stability(shard1_summary=within, cross_summary=cross, mixed_summary=mixed)
    assert within["proxy_beats_current_fraction"] == 1.0
    assert within["proxy_beats_random_fraction"] == 1.0
    assert within["full_context_noise_confirmed"] is True
    assert decision["proxy_signal_stable_across_shards"] is True


def test_skipped_cross_shard_has_reason():
    skipped = skipped_eval_summary("cross_shard", "missing shard0")
    assert skipped["skipped"] is True
    assert skipped["skip_reason"] == "missing shard0"
