import torch

from data.bridgedata_v2_factorized_selector_dataset_step37 import (
    batch_step37_samples,
    make_step37_sample_from_tensors,
    validate_step37_sample,
)


def test_step37_fake_dataset_builds_patch_and_temporal_targets_and_hides_future():
    sample = make_step37_sample_from_tensors(
        sample_id="s0",
        trajectory_id="traj0",
        shard_id="shard1",
        data_package_id="pkg",
        context_tokens=torch.randn(4, 3, 6),
        current_tokens=torch.randn(2, 3, 6),
        proxy_patch_target=torch.rand(4, 3),
        proxy_temporal_target=torch.rand(4),
    )
    assert validate_step37_sample(sample) is True
    assert list(sample["context_tokens"].shape) == [4, 3, 6]
    assert list(sample["current_tokens"].shape) == [2, 3, 6]
    assert list(sample["proxy_patch_target"].shape) == [4, 3]
    assert list(sample["proxy_temporal_target"].shape) == [4]
    assert sample["metadata"]["future_tokens_exposed_to_selector"] is False
    assert sample["metadata"]["action_used_as_input"] is False
    assert sample["metadata"]["language_used_as_input"] is False


def test_step37_batch_stacks_tokens_and_targets_to_device():
    sample = make_step37_sample_from_tensors(
        sample_id="s0",
        trajectory_id="traj0",
        shard_id="shard0",
        data_package_id="pkg",
        context_tokens=torch.randn(4, 3, 6),
        current_tokens=torch.randn(2, 3, 6),
        proxy_patch_target=torch.rand(4, 3),
        proxy_temporal_target=torch.rand(4),
    )
    batch = batch_step37_samples([sample, sample], torch.device("cpu"))
    assert list(batch["context_tokens"].shape) == [2, 4, 3, 6]
    assert list(batch["current_tokens"].shape) == [2, 2, 3, 6]
    assert list(batch["proxy_patch_target"].shape) == [2, 4, 3]
    assert list(batch["proxy_temporal_target"].shape) == [2, 4]
