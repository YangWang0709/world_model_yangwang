import yaml
import torch

from training.bridgedata_v2_tfds_tiny_overfit_trainer import train_tiny_overfit


def _fake_sample():
    return {
        "sample_id": "sample",
        "trajectory_id": "traj",
        "context_tokens": torch.zeros((16, 392, 768), dtype=torch.float32),
        "current_tokens": torch.zeros((4, 392, 768), dtype=torch.float32),
        "future_tokens": torch.zeros((4, 392, 768), dtype=torch.float32),
        "context_importance": torch.ones((16, 392), dtype=torch.float32),
        "metadata": {
            "current_tokens_kept_full": True,
            "train_current_importance": False,
            "action_used_as_input": False,
            "language_used_as_input": False,
            "goal_used_as_input": False,
        },
    }


def test_tiny_overfit_summary_keeps_forbidden_training_flags_false(tmp_path):
    config = yaml.safe_load(
        open("configs/bridgedata_v2_tfds_world_model_tiny_overfit_step27.yaml", encoding="utf-8")
    )
    config["policies"] = [{"name": "current_only", "use_context": False, "topk": 0, "train": True}]
    config["tiny_overfit"].update(
        {"hidden_dim": 16, "train_steps": 5, "eval_every": 5, "learning_rate": 0.01, "device": "cpu"}
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
    payload = train_tiny_overfit(config, samples=[_fake_sample()], write_outputs=False)
    summary = payload["summary"]
    assert summary["download_performed"] is False
    assert summary["model_download_performed"] is False
    assert summary["token_extraction_performed"] is False
    assert summary["importance_generation_performed"] is False
    assert summary["videomae_training_performed"] is False
    assert summary["teacher_training_performed"] is False
    assert summary["selector_training_performed"] is False
    assert summary["current_importance_training_performed"] is False
    assert summary["world_model_large_training_performed"] is False
    assert summary["data_token_shards_written"] is False
    assert summary["data_importance_shards_written"] is False
    assert summary["checkpoint_saved"] is False
    assert summary["optimizer_step_scope"] == "tiny_world_model_predictor_only"
