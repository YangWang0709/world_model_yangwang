import torch

from data.bridgedata_v2_proxy_temporal_selector_dataset_step35 import make_step35_sample_from_tensors
from training.bridgedata_v2_proxy_temporal_selector_trainer import _train_one_setting


def test_step35_trainer_updates_only_proxy_temporal_selector_head():
    config = {
        "seed": 7,
        "selector": {
            "token_dim": 6,
            "hidden_dim": 8,
            "context_frames": 4,
            "spatial_tokens": 3,
            "condition_on_current_summary": True,
            "dropout": 0.0,
            "detach_token_inputs": True,
        },
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "train_steps": 3,
            "eval_every": 1,
            "batch_size": 2,
            "grad_clip_norm": 1.0,
        },
        "loss": {"mse_weight": 1.0, "rank_weight": 0.1, "topk_soft_weight": 0.1, "topk_frames": [1, 2]},
    }
    samples = [_sample(f"s{i}", "traj_train" if i < 3 else "traj_val") for i in range(5)]
    samples_by_shard = {"shard1": samples}
    setting = {
        "name": "fake_within",
        "eval_type": "within_shard",
        "split_seed": 42,
        "train_sample_ids_by_shard": {"shard1": ["s0", "s1", "s2"]},
        "val_sample_ids_by_shard": {"shard1": ["s3", "s4"]},
    }
    row, curve = _train_one_setting(config, setting, samples_by_shard, torch.device("cpu"))
    assert row["optimizer_step_performed"] is True
    assert row["optimizer_scope"] == "proxy_temporal_selector_head_only"
    assert row["videomae_training_performed"] is False
    assert row["current_importance_training_performed"] is False
    assert row["final_selector_training_performed"] is False
    assert row["patch_level_selector_training_performed"] is False
    assert row["checkpoint_saved"] is False
    assert curve["curve"]


def _sample(sample_id: str, trajectory_id: str):
    return make_step35_sample_from_tensors(
        sample_id=sample_id,
        trajectory_id=trajectory_id,
        shard_id="shard1",
        data_package_id="fake",
        context_tokens=torch.randn(4, 3, 6),
        current_tokens=torch.randn(2, 3, 6),
        proxy_temporal_target=torch.rand(4),
    )
