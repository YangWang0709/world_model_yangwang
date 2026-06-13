import torch

from data.bridgedata_v2_current_conditioning_dataset_step41a import (
    attach_step41a_targets,
    batch_step41a_samples,
    build_step41a_target_label,
    build_step41a_training_settings_for_fake,
    build_train_split_global_spatial_prior,
    make_step41a_sample_from_tensors,
    validate_step41a_sample,
)


def test_step41a_fake_dataset_builds_train_prior_targets_and_batches_without_grad():
    samples = [_sample(f"s{i}") for i in range(3)]
    prior_payload = build_train_split_global_spatial_prior(samples[:2])
    train_prior = prior_payload["prior"]
    target = build_step41a_target_label(samples[0]["original_importance"], train_prior)
    attached = attach_step41a_targets(samples[:2], train_prior)
    batch = batch_step41a_samples(attached, torch.device("cpu"))

    assert prior_payload["stats"]["built_from_train_split_only"] is True
    assert prior_payload["stats"]["val_split_used_for_prior"] is False
    assert list(target.shape) == [4, 5]
    assert float(target.min()) >= 0.0
    assert float(target.max()) <= 1.0
    assert list(batch["context_tokens"].shape) == [2, 4, 5, 6]
    assert list(batch["current_tokens"].shape) == [2, 2, 5, 6]
    assert list(batch["target_label"].shape) == [2, 4, 5]
    assert batch["context_tokens"].dtype == torch.float32
    assert batch["current_tokens"].requires_grad is False
    assert batch["target_label"].requires_grad is False
    assert validate_step41a_sample(samples[0]) is True


def test_step41a_fake_split_settings_are_shard_aware_and_disjoint():
    settings = build_step41a_training_settings_for_fake()

    assert {setting["eval_type"] for setting in settings} == {"within_shard", "cross_shard", "mixed_shard"}
    assert all(setting["strict_shard_aware_split"] for setting in settings)
    assert all(setting["train_val_sample_id_disjoint"] for setting in settings)


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
