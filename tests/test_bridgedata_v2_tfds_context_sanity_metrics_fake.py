import yaml

from analysis.bridgedata_v2_tfds_context_sanity import build_context_sanity_diagnostics


def _config():
    return yaml.safe_load(open("configs/bridgedata_v2_tfds_context_sanity_step28.yaml", encoding="utf-8"))


def _artifacts():
    return {
        "step24_summary": {"token_extraction_performed": True, "safe_stop": False},
        "step25_summary": {
            "importance_generation_performed": True,
            "safe_stop": False,
            "num_samples": 4,
            "method": "proxy_token_mse_dryrun",
            "importance_raw_mean": 5.0e-6,
            "importance_raw_std": 8.0e-6,
            "importance_norm_min": 0.0,
            "importance_norm_max": 1.0,
            "label_quality_note": "proxy dry-run only; not final teacher label",
        },
        "step26_summary": {"world_model_smoke_performed": True, "safe_stop": False, "num_samples": 4},
        "step26_policy_metrics": {
            "policy_metrics": [
                {"policy": "current_only", "mean_selected_importance_mass": 0.0},
                {"policy": "random_context_topk", "mean_selected_importance_mass": 0.037160635033425635},
                {"policy": "proxy_importance_topk", "mean_selected_importance_mass": 0.2452614637082408},
                {"policy": "full_context_reference", "mean_selected_importance_mass": 1.0},
            ]
        },
        "step27_summary": {
            "tiny_overfit_training_performed": True,
            "safe_stop": False,
            "num_samples": 4,
            "loss_quality_note": "tiny-overfit only; not final performance",
        },
        "step27_policy_comparison": {
            "policy_metrics": [
                {
                    "policy": "current_only",
                    "initial_loss": 12.137228012084961,
                    "final_loss": 0.0035450139548629522,
                    "best_loss": 0.0011004924308508635,
                    "relative_loss_decrease": 0.9997079222742349,
                },
                {
                    "policy": "random_context_topk",
                    "initial_loss": 12.773147583007812,
                    "final_loss": 0.0021274862810969353,
                    "best_loss": 0.0021274862810969353,
                    "relative_loss_decrease": 0.9998334407187209,
                },
                {
                    "policy": "proxy_importance_topk",
                    "initial_loss": 12.673670768737793,
                    "final_loss": 0.07361903041601181,
                    "best_loss": 0.0013253530487418175,
                    "relative_loss_decrease": 0.9941911832996634,
                },
                {
                    "policy": "full_context_reference",
                    "initial_loss": 12.252837181091309,
                    "final_loss": 0.0011100443080067635,
                    "best_loss": 0.0011100443080067635,
                    "relative_loss_decrease": 0.9999094051204958,
                },
            ]
        },
    }


def test_context_sanity_metrics_detect_memorization_and_proxy_limits():
    payload = build_context_sanity_diagnostics(_config(), _artifacts())
    summary = payload["summary"]
    context = payload["context_utility"]
    memorization = payload["memorization_risk"]
    importance = payload["importance_label"]

    assert summary["current_only_can_overfit"] is True
    assert summary["memorization_risk"] == "high"
    assert memorization["requires_train_val_split"] is True
    assert context["proxy_topk_selects_concentrated_context"] is True
    assert context["proxy_importance_mass_advantage_over_random"] > 6.0
    assert context["proxy_final_loss_better_than_random"] is False
    assert context["proxy_best_loss_close_to_full"] is True
    assert context["context_utility_claim_allowed"] is False
    assert importance["label_is_proxy"] is True
    assert importance["label_is_final_teacher"] is False
    assert importance["raw_signal_small"] is True
    assert summary["train_current_importance_now"] is False

