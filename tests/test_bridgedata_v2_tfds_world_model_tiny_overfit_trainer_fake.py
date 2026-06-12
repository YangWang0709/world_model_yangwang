import torch
import yaml

from training.bridgedata_v2_tfds_tiny_overfit_trainer import train_tiny_overfit


def _fake_sample(index: int):
    value = float(index + 1) / 10.0
    return {
        "sample_id": f"sample_{index}",
        "trajectory_id": f"traj_{index}",
        "context_tokens": torch.full((16, 392, 768), value, dtype=torch.float32),
        "current_tokens": torch.full((4, 392, 768), value * 2.0, dtype=torch.float32),
        "future_tokens": torch.zeros((4, 392, 768), dtype=torch.float32),
        "context_importance": torch.linspace(0.0, 1.0, 16 * 392, dtype=torch.float32).reshape(16, 392),
        "metadata": {
            "current_tokens_kept_full": True,
            "train_current_importance": False,
            "action_used_as_input": False,
            "language_used_as_input": False,
            "goal_used_as_input": False,
        },
    }


def _tiny_config(tmp_path):
    config = yaml.safe_load(
        open("configs/bridgedata_v2_tfds_world_model_tiny_overfit_step27.yaml", encoding="utf-8")
    )
    config["policies"] = [
        {"name": "current_only", "use_context": False, "topk": 0, "train": True},
        {"name": "proxy_importance_topk", "use_context": True, "selection": "importance_topk", "topk": 32, "train": True},
    ]
    config["tiny_overfit"].update(
        {
            "hidden_dim": 32,
            "train_steps": 25,
            "eval_every": 5,
            "learning_rate": 0.01,
            "device": "cpu",
        }
    )
    config["acceptance"]["min_policies_with_loss_decrease"] = 1
    for key in [
        "training_summary_json",
        "training_summary_md",
        "loss_curves_json",
        "policy_comparison_json",
        "final_eval_json",
        "eval_json",
        "eval_md",
    ]:
        suffix = ".md" if key.endswith("_md") else ".json"
        config["output"][key] = str(tmp_path / f"{key}{suffix}")
    return config


def test_fake_samples_train_tiny_predictor_only_and_loss_decreases(tmp_path):
    samples = [_fake_sample(i) for i in range(4)]
    payload = train_tiny_overfit(_tiny_config(tmp_path), samples=samples, write_outputs=True)
    summary = payload["summary"]
    assert summary["tiny_overfit_training_performed"] is True
    assert summary["training_performed"] is True
    assert summary["optimizer_step_performed"] is True
    assert summary["optimizer_step_scope"] == "tiny_world_model_predictor_only"
    assert summary["optimizer_param_count"] == summary["tiny_predictor_param_count"]
    assert summary["all_losses_finite"] is True
    assert summary["policies_with_loss_decrease"] >= 1
    assert summary["current_tokens_kept_full"] is True
    assert summary["train_current_importance"] is False
    assert summary["current_importance_training_performed"] is False
    assert summary["videomae_training_performed"] is False
    assert summary["teacher_training_performed"] is False
    assert summary["selector_training_performed"] is False
    assert summary["checkpoint_saved"] is False
    assert (tmp_path / "training_summary_json.json").exists()
    for sample in samples:
        assert sample["current_tokens"].requires_grad is False
        assert sample["context_tokens"].requires_grad is False
        assert sample["future_tokens"].requires_grad is False
        assert sample["context_importance"].requires_grad is False
