from analysis.bridgedata_v2_current_dominance_diagnosis import diagnose_current_dominance


def test_step31_current_dominance_keeps_current_training_disabled():
    result = diagnose_current_dominance(
        {
            "target_rows": [
                {"proxy_gain_over_current": 0.02, "full_context_noise_penalty_present": True, "current_only_val": 3.0},
                {"proxy_gain_over_current": 0.03, "full_context_noise_penalty_present": True, "current_only_val": 3.2},
            ]
        }
    )
    assert result["current_dominance_level"] == "high"
    assert result["train_current_importance_now"] is False
    assert result["current_importance_training_allowed"] is False
