import torch

from data.bridgedata_v2_current_conditioning_dataset_step41a import make_step41a_sample_from_tensors
from training.bridgedata_v2_current_conditioning_trainer_step41a import _train_one_setting_variant


def test_step41a_trainer_updates_only_current_conditioned_selector_head():
    config = {
        "seed": 7,
        "label": {"variant": "global_spatial_prior_removed_residual"},
        "data": {"token_dim": 6, "context_frames": 4, "current_frames": 2, "spatial_tokens": 5},
        "model": {
            "token_dim": 6,
            "hidden_dim": 8,
            "context_frames": 4,
            "current_frames": 2,
            "spatial_tokens": 5,
            "dropout": 0.0,
        },
        "current_conditioning_variants": {
            "attention_dim": 4,
            "coarse_current_bins": 3,
            "chunk_context_tokens": 4,
            "max_attention_elements_per_batch": 1000,
            "attention_temperature": 1.0,
            "detach_token_inputs": True,
            "use_temporal_embedding": True,
            "use_spatial_embedding": True,
        },
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "train_steps": 2,
            "eval_every": 1,
            "batch_size": 2,
            "eval_batch_size": 2,
            "grad_clip_norm": 1.0,
        },
        "loss": {"use_sigmoid_scores": True},
        "metrics": {"topk_values": [2, 4]},
    }
    samples = [_sample(f"s{i}") for i in range(4)]
    setting = {
        "name": "fake_within",
        "eval_type": "within_shard",
        "split_seed": 42,
        "train_sample_ids_by_shard": {"shard1": ["s0", "s1"]},
        "val_sample_ids_by_shard": {"shard1": ["s2", "s3"]},
    }

    row, curve = _train_one_setting_variant(
        config,
        setting,
        {"shard1": samples},
        torch.device("cpu"),
        "current_coarse_spatial_query_attention",
    )

    assert row["optimizer_step_performed"] is True
    assert row["optimizer_scope"] == "current_conditioned_selector_head_only"
    assert row["videomae_loaded"] is False
    assert row["videomae_training_performed"] is False
    assert row["current_importance_training_performed"] is False
    assert row["final_selector_training_performed"] is False
    assert row["world_model_training_performed"] is False
    assert row["downstream_task_training_performed"] is False
    assert row["checkpoint_saved"] is False
    assert row["state_dict_saved"] is False
    assert row["max_attention_elements_seen"] > 0
    assert curve["curve"]


def _sample(sample_id: str):
    return make_step41a_sample_from_tensors(
        sample_id=sample_id,
        trajectory_id=f"traj_{sample_id}",
        shard_id="shard1",
        data_package_id="fake",
        context_tokens=torch.randn(4, 5, 6),
        current_tokens=torch.randn(2, 5, 6),
        original_importance=torch.rand(4, 5),
    )
