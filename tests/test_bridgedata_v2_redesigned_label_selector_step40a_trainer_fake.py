import torch

from data.bridgedata_v2_redesigned_label_selector_dataset_step40a import make_step40a_sample_from_tensors
from training.bridgedata_v2_redesigned_label_selector_trainer_step40a import _train_one_setting


def test_step40a_trainer_updates_only_redesigned_label_selector_head():
    config = {
        "seed": 7,
        "label": {"variant": "global_spatial_prior_removed_residual"},
        "model": {
            "token_dim": 6,
            "hidden_dim": 8,
            "context_frames": 4,
            "spatial_tokens": 5,
            "condition_on_current_summary": True,
            "use_temporal_embedding": True,
            "use_spatial_embedding": True,
            "dropout": 0.0,
            "detach_token_inputs": True,
        },
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "train_steps": 3,
            "eval_every": 1,
            "batch_size": 2,
            "eval_batch_size": 2,
            "grad_clip_norm": 1.0,
        },
        "loss": {"use_sigmoid_scores": True},
        "metrics": {"topk_values": [64, 128, 256, 512]},
    }
    samples = [_sample(f"s{i}") for i in range(5)]
    setting = {
        "name": "fake_within",
        "eval_type": "within_shard",
        "split_seed": 42,
        "train_sample_ids_by_shard": {"shard1": ["s0", "s1", "s2"]},
        "val_sample_ids_by_shard": {"shard1": ["s3", "s4"]},
    }

    row, curve = _train_one_setting(config, setting, {"shard1": samples}, torch.device("cpu"))

    assert row["optimizer_step_performed"] is True
    assert row["optimizer_scope"] == "redesigned_label_proxy_selector_head_only"
    assert row["videomae_loaded"] is False
    assert row["videomae_training_performed"] is False
    assert row["current_importance_training_performed"] is False
    assert row["final_selector_training_performed"] is False
    assert row["world_model_training_performed"] is False
    assert row["downstream_task_training_performed"] is False
    assert row["checkpoint_saved"] is False
    assert row["state_dict_saved"] is False
    assert curve["curve"]


def _sample(sample_id: str):
    return make_step40a_sample_from_tensors(
        sample_id=sample_id,
        trajectory_id=f"traj_{sample_id}",
        shard_id="shard1",
        data_package_id="fake",
        context_tokens=torch.randn(4, 5, 6),
        current_tokens=torch.randn(2, 5, 6),
        original_importance=torch.rand(4, 5),
    )
