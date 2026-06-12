from analysis.bridgedata_v2_temporal_horizon_diagnosis import summarize_target_row, summarize_temporal_horizon_results


def test_step31_temporal_horizon_summary_fake():
    row = summarize_target_row(
        "future_last",
        [
            {"split_seed": 42, "current_only": 4.0, "proxy_importance_topk": 3.0, "full_context_reference": 4.2},
            {"split_seed": 123, "current_only": 5.0, "proxy_importance_topk": 4.5, "full_context_reference": 4.9},
        ],
    )
    assert row["proxy_gain_over_current"] > 0.0
    assert row["full_context_noise_penalty_present"] is True
    summary = summarize_temporal_horizon_results([row])
    assert summary["temporal_horizon_diagnosis_performed"] is True
    assert summary["best_target_variant"] == "future_last"
    assert summary["token_extraction_performed"] is False
