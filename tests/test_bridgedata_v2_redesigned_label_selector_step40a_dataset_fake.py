import torch

from data.bridgedata_v2_redesigned_label_selector_dataset_step40a import (
    attach_targets,
    batch_step40a_samples,
    build_global_spatial_prior_removed_residual_target,
    build_train_split_global_spatial_prior,
    make_step40a_sample_from_tensors,
)


def test_step40a_fake_samples_build_train_only_prior_and_targets():
    train = [_sample(f"train{i}", scale=1.0 + i * 0.1) for i in range(3)]
    val = [_sample(f"val{i}", scale=3.0 + i) for i in range(2)]

    train_prior = build_train_split_global_spatial_prior(train)
    val_prior = build_train_split_global_spatial_prior(val)
    target = build_global_spatial_prior_removed_residual_target(train[0]["original_importance"], train_prior["prior"])

    assert list(train_prior["prior"].shape) == [5]
    assert train_prior["stats"]["built_from_train_split_only"] is True
    assert train_prior["stats"]["val_split_used_for_prior"] is False
    assert not torch.allclose(train_prior["prior"], val_prior["prior"])
    assert list(target.shape) == [4, 5]
    assert float(target.min()) >= 0.0
    assert float(target.max()) <= 1.0


def test_step40a_batch_contains_only_tokens_and_target_label():
    samples = attach_targets([_sample("s0"), _sample("s1")], torch.rand(5))

    batch = batch_step40a_samples(samples, torch.device("cpu"))

    assert list(batch["context_tokens"].shape) == [2, 4, 5, 6]
    assert list(batch["current_tokens"].shape) == [2, 2, 5, 6]
    assert list(batch["target_label"].shape) == [2, 4, 5]


def _sample(sample_id: str, *, scale: float = 1.0):
    return make_step40a_sample_from_tensors(
        sample_id=sample_id,
        trajectory_id=f"traj_{sample_id}",
        shard_id="shard1",
        data_package_id="fake",
        context_tokens=torch.randn(4, 5, 6),
        current_tokens=torch.randn(2, 5, 6),
        original_importance=torch.rand(4, 5) * float(scale),
    )
