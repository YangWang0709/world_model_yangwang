import torch

from data.bridgedata_v2_proxy_patch_selector_dataset_step36 import (
    batch_step36_samples,
    make_step36_sample_from_tensors,
    validate_step36_sample,
)


def test_step36_fake_dataset_builds_patch_target_and_hides_future():
    context = torch.arange(4 * 3 * 6, dtype=torch.float32).reshape(4, 3, 6)
    current = torch.ones(2, 3, 6)
    target = torch.rand(4, 3)
    sample = make_step36_sample_from_tensors(
        sample_id="s0",
        trajectory_id="traj0",
        shard_id="shard1",
        data_package_id="pkg",
        context_tokens=context,
        current_tokens=current,
        proxy_patch_target=target,
    )
    assert validate_step36_sample(sample) is True
    assert list(sample["context_tokens"].shape) == [4, 3, 6]
    assert list(sample["current_tokens"].shape) == [2, 3, 6]
    assert list(sample["proxy_patch_target"].shape) == [4, 3]
    assert sample["metadata"]["future_tokens_exposed_to_selector"] is False
    assert sample["metadata"]["action_used_as_input"] is False
    assert sample["metadata"]["language_used_as_input"] is False


def test_step36_batch_stacks_cpu_tokens_to_device():
    sample = make_step36_sample_from_tensors(
        sample_id="s0",
        trajectory_id="traj0",
        shard_id="shard0",
        data_package_id="pkg",
        context_tokens=torch.randn(4, 3, 6),
        current_tokens=torch.randn(2, 3, 6),
        proxy_patch_target=torch.rand(4, 3),
    )
    batch = batch_step36_samples([sample, sample], torch.device("cpu"))
    assert list(batch["context_tokens"].shape) == [2, 4, 3, 6]
    assert list(batch["current_tokens"].shape) == [2, 2, 3, 6]
    assert list(batch["proxy_patch_target"].shape) == [2, 4, 3]
