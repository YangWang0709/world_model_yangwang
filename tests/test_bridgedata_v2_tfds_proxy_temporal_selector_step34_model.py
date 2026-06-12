import torch
import torch.nn.functional as F

from models.bridgedata_v2_proxy_temporal_selector import (
    ProxyTemporalSelectorHead,
    proxy_temporal_selector_dry_run,
    proxy_temporal_targets,
)


def test_proxy_temporal_selector_detaches_token_inputs_but_trains_head():
    selector = ProxyTemporalSelectorHead(
        token_dim=4,
        hidden_dim=8,
        context_frames=2,
        spatial_tokens=3,
        condition_on_current_summary=True,
        detach_token_inputs=True,
    )
    context = torch.randn(1, 2, 3, 4, requires_grad=True)
    current = torch.randn(1, 2, 3, 4, requires_grad=True)
    target = torch.tensor([[[0.0, 1.0, 0.5], [1.0, 0.0, 0.25]]])

    scores = selector(context, current)
    loss = F.mse_loss(torch.sigmoid(scores), proxy_temporal_targets(target))
    loss.backward()

    assert list(scores.shape) == [1, 2]
    assert context.grad is None
    assert current.grad is None
    assert any(param.grad is not None for param in selector.parameters())


def test_proxy_temporal_target_shape_and_normalization():
    importance = torch.tensor([[1.0, 1.0, 1.0], [2.0, 4.0, 6.0]])
    target = proxy_temporal_targets(importance)
    assert list(target.shape) == [1, 2]
    assert float(target.min()) >= 0.0
    assert float(target.max()) <= 1.0


def test_proxy_temporal_selector_dry_run_is_no_optimizer_step():
    selector = ProxyTemporalSelectorHead(
        token_dim=4,
        hidden_dim=8,
        context_frames=2,
        spatial_tokens=3,
        condition_on_current_summary=True,
    )
    sample = {
        "sample_id": "fake",
        "trajectory_id": "traj",
        "context_tokens": torch.randn(2, 3, 4),
        "current_tokens": torch.randn(2, 3, 4),
        "context_importance": torch.rand(2, 3),
    }
    summary = proxy_temporal_selector_dry_run(selector=selector, samples=[sample])
    assert summary["selector_forward_performed"] is True
    assert summary["selector_training_performed"] is False
    assert summary["optimizer_step_performed"] is False
    assert summary["checkpoint_saved"] is False
    assert summary["videomae_frozen"] is True
    assert summary["current_importance_training_performed"] is False
    assert summary["all_losses_finite"] is True
