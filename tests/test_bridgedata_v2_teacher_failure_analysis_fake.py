from analysis.bridgedata_v2_teacher_failure_analysis import build_teacher_failure_analysis


def test_step31_failure_analysis_recommends_without_selector_training():
    result = build_teacher_failure_analysis(
        {"best_architecture_variant": "current_conditioned_attention_with_proxy_prior", "proxy_prior_helped_attention": True},
        {"best_diagnostic_score": "proxy_importance", "teacher_topk_underperforms_proxy_confirmed": True},
        {"full_context_noise_confirmed": True, "delta_target_increases_context_gain": False},
        {"current_dominance_level": "medium"},
        {"teacher_proxy_pearson_mean": -0.006},
    )
    assert result["teacher_topk_underperforms_proxy_confirmed"] is True
    assert result["recommended_step32"]
    assert result["selector_training_allowed"] is False
    assert result["current_importance_training_allowed"] is False
