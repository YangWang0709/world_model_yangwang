import torch

from data.bridgedata_v2_factorized_selector_dataset_step37 import (
    OPTIMIZER_SCOPE,
    make_step37_sample_from_tensors,
)
from training.bridgedata_v2_factorized_selector_trainer_step37 import _train_one_setting


def test_step37_trainer_updates_only_factorized_selector_head():
    config = {
        "seed": 7,
        "factorized_selector": {
            "token_dim": 6,
            "hidden_dim": 8,
            "context_frames": 4,
            "spatial_tokens": 5,
            "condition_on_current_summary": True,
            "use_temporal_embedding": True,
            "use_spatial_embedding": True,
            "use_temporal_branch": True,
            "use_spatial_residual_branch": True,
            "combine_mode": "temporal_plus_spatial_residual",
            "detach_token_inputs": True,
            "proxy_temporal_aux_weight": 0.1,
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
        "loss_ablation": {
            "rank_pairs_per_batch": 16,
            "topk_values": [2, 4],
        },
        "metrics": {"topk_values": [64, 128, 256, 512]},
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
    row, curve = _train_one_setting(
        config,
        setting,
        samples_by_shard,
        torch.device("cpu"),
        loss_variant="mse_plus_rank_plus_topk_soft",
        use_proxy_temporal_prior=True,
        optimizer_scope=OPTIMIZER_SCOPE,
        step36_reference_mse=0.2,
    )
    assert row["optimizer_step_performed"] is True
    assert row["optimizer_scope"] == "proxy_factorized_selector_head_only"
    assert row["videomae_training_performed"] is False
    assert row["current_importance_training_performed"] is False
    assert row["final_selector_training_performed"] is False
    assert row["factorized_selector_head_training_performed"] is True
    assert row["checkpoint_saved"] is False
    assert curve["curve"]


def _sample(sample_id: str, trajectory_id: str):
    return make_step37_sample_from_tensors(
        sample_id=sample_id,
        trajectory_id=trajectory_id,
        shard_id="shard1",
        data_package_id="fake",
        context_tokens=torch.randn(4, 5, 6),
        current_tokens=torch.randn(2, 5, 6),
        proxy_patch_target=torch.rand(4, 5),
        proxy_temporal_target=torch.rand(4),
    )
