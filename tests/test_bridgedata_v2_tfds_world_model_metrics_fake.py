from data.bridgedata_v2_tfds_world_model_smoke_metrics import (
    LOSS_QUALITY_NOTE,
    policy_metric_table,
    summarize_forward_loss,
    summarize_policy_metrics,
)


def test_policy_metrics_and_forward_summary_are_computed():
    records = [
        {
            "policy": "current_only",
            "topk": 0,
            "loss": 1.0,
            "loss_finite": True,
            "selected_importance_mass": 0.0,
            "selected_importance_mean": 0.0,
            "current_tokens_kept_full": True,
            "optimizer_step_performed": False,
            "training_performed": False,
        },
        {
            "policy": "proxy_importance_topk",
            "topk": 256,
            "loss": 0.5,
            "loss_finite": True,
            "selected_importance_mass": 0.4,
            "selected_importance_mean": 0.2,
            "current_tokens_kept_full": True,
            "optimizer_step_performed": False,
            "training_performed": False,
        },
    ]
    metrics = summarize_policy_metrics(records)
    table = policy_metric_table(metrics)
    forward = summarize_forward_loss(records)
    assert forward["all_losses_finite"] is True
    assert forward["loss_quality_note"] == LOSS_QUALITY_NOTE
    assert table[0]["policy"] == "current_only"
    assert any(item["policy"] == "proxy_importance_topk" and item["mean_selected_importance_mass"] == 0.4 for item in table)
