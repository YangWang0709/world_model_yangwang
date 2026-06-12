from data.bridgedata_v2_tfds_teacher_topk_metrics import (
    recommended_step31_from_teacher_topk,
    summarize_teacher_topk_utility,
)


def _metric(policy, loss):
    return {"policy": policy, "val_final_loss": loss}


def test_teacher_topk_metrics_compare_all_policies_and_keep_claims_disabled():
    summary = summarize_teacher_topk_utility(
        [
            {
                "split_seed": 42,
                "policy_metrics": [
                    _metric("current_only", 4.0),
                    _metric("random_context_topk", 4.1),
                    _metric("proxy_importance_topk", 3.8),
                    _metric("teacher_occlusion_topk", 3.7),
                    _metric("full_context_reference", 4.4),
                ],
            },
            {
                "split_seed": 123,
                "policy_metrics": [
                    _metric("current_only", 4.0),
                    _metric("random_context_topk", 4.1),
                    _metric("proxy_importance_topk", 3.7),
                    _metric("teacher_occlusion_topk", 3.9),
                    _metric("full_context_reference", 4.4),
                ],
            },
        ]
    )
    assert summary["teacher_topk_utility_performed"] is True
    assert summary["teacher_beats_current_fraction"] == 1.0
    assert summary["teacher_beats_random_fraction"] == 1.0
    assert summary["teacher_beats_proxy_fraction"] == 0.5
    assert summary["context_utility_claim_allowed"] is False
    assert summary["selector_training_allowed"] is False
    assert summary["current_importance_training_allowed"] is False
    assert recommended_step31_from_teacher_topk(summary)["name"]
