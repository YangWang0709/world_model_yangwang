import torch

from data.bridgedata_v2_proxy_temporal_selector_dataset_step35 import (
    batch_step35_samples,
    make_step35_sample_from_tensors,
    validate_step35_sample,
)
from models.bridgedata_v2_proxy_temporal_selector import proxy_temporal_targets


def test_step35_fake_dataset_builds_temporal_target_and_hides_future():
    context = torch.arange(4 * 3 * 6, dtype=torch.float32).reshape(4, 3, 6)
    current = torch.ones(2, 3, 6)
    importance = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
        ]
    )
    target = proxy_temporal_targets(importance).squeeze(0)
    sample = make_step35_sample_from_tensors(
        sample_id="s0",
        trajectory_id="traj0",
        shard_id="shard1",
        data_package_id="pkg",
        context_tokens=context,
        current_tokens=current,
        proxy_temporal_target=target,
    )
    assert validate_step35_sample(sample) is True
    assert list(sample["context_temporal_features"].shape) == [4, 6]
    assert list(sample["current_summary"].shape) == [6]
    assert list(sample["proxy_temporal_target"].shape) == [4]
    assert sample["metadata"]["future_tokens_exposed_to_selector"] is False
    assert sample["metadata"]["action_used_as_input"] is False
    assert sample["metadata"]["language_used_as_input"] is False


def test_step35_batch_stacks_cpu_features_to_device():
    sample = make_step35_sample_from_tensors(
        sample_id="s0",
        trajectory_id="traj0",
        shard_id="shard0",
        data_package_id="pkg",
        context_tokens=torch.randn(4, 3, 6),
        current_tokens=torch.randn(2, 3, 6),
        proxy_temporal_target=torch.rand(4),
    )
    batch = batch_step35_samples([sample, sample], torch.device("cpu"))
    assert list(batch["context_temporal_features"].shape) == [2, 4, 6]
    assert list(batch["current_summary"].shape) == [2, 6]
    assert list(batch["proxy_temporal_target"].shape) == [2, 4]
