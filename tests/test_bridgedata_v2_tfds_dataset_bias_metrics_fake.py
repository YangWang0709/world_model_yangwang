from data.bridgedata_v2_tfds_dataset_bias_metrics import summarize_dataset_bias


def test_dataset_bias_detects_length_and_proxy_shift_without_raw_language():
    summary = summarize_dataset_bias(
        shard0_summary={"length_stats": {"mean": 10.0}, "action_stats": {"mean_abs_mean": 1.0}, "language_hash_counts": {"a": 2}},
        shard1_summary={"length_stats": {"mean": 20.0}, "action_stats": {"mean_abs_mean": 3.0}, "language_hash_counts": {"b": 2}},
        shard0_importance_summary={"topk_mass_mean": 0.2, "temporal_concentration_mean": 0.1},
        shard1_importance_summary={"topk_mass_mean": 0.6, "temporal_concentration_mean": 0.4},
    )
    assert summary["dataset_bias_detected"] is True
    assert summary["distribution_shift_flags"]["episode_length_shift"] is True
    assert summary["distribution_shift_flags"]["language_hash_shift"] is True
    assert summary["raw_language_text_saved"] is False


def test_dataset_bias_can_report_no_shift():
    summary = summarize_dataset_bias(
        shard0_summary={"length_stats": {"mean": 10.0}, "action_stats": {"mean_abs_mean": 1.0}, "language_hash_counts": {"a": 2}},
        shard1_summary={"length_stats": {"mean": 10.1}, "action_stats": {"mean_abs_mean": 1.1}, "language_hash_counts": {"a": 2}},
        shard0_importance_summary={"topk_mass_mean": 0.2, "temporal_concentration_mean": 0.1},
        shard1_importance_summary={"topk_mass_mean": 0.21, "temporal_concentration_mean": 0.11},
    )
    assert summary["dataset_bias_detected"] is False
