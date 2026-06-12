from analysis.bridgedata_v2_tfds_teacher_planning import build_next_step_decision, build_teacher_plan


def test_teacher_plan_has_required_candidates_and_keeps_current_importance_off():
    plan = build_teacher_plan(sample_count=4)
    names = [candidate["name"] for candidate in plan["candidate_plans"]]
    assert len(names) >= 4
    assert "stronger trained predictor occlusion teacher" in names
    assert "train/val context utility experiment first" in names
    assert "true temporal VideoMAE token extraction" in names
    assert "current importance diagnostic only" in names
    assert plan["train_current_importance_now"] is False
    assert plan["do_not_train_current_importance_yet"] is True


def test_next_step_decision_is_scale_first_with_teacher_alternative():
    decision = build_next_step_decision(sample_count=4)
    assert "32/64-window" in decision["recommended_step29"]["name"]
    assert "existing shard" in decision["recommended_step29"]["scope"]
    assert "occlusion teacher" in decision["alternative_step29"]["name"]
    assert decision["do_not_train_current_importance_yet"] is True
    assert decision["do_not_claim_context_utility_from_4_sample_overfit"] is True

