import torch

from models.bridgedata_v2_proxy_patch_token_selector import ProxyPatchTokenSelectorHead, proxy_patch_targets


def test_step36_patch_selector_forward_outputs_patch_scores_and_detaches_inputs():
    model = ProxyPatchTokenSelectorHead(
        token_dim=6,
        hidden_dim=8,
        context_frames=4,
        spatial_tokens=3,
        condition_on_current_summary=True,
        use_temporal_embedding=True,
        use_spatial_embedding=True,
        detach_token_inputs=True,
    )
    context = torch.randn(2, 4, 3, 6, requires_grad=True)
    current = torch.randn(2, 2, 3, 6, requires_grad=True)
    scores = model(context, current)
    assert list(scores.shape) == [2, 4, 3]
    scores.mean().backward()
    assert context.grad is None
    assert current.grad is None
    assert any(param.grad is not None for param in model.parameters())


def test_step36_patch_targets_accept_unbatched_context_importance():
    target = proxy_patch_targets(torch.rand(4, 3))
    assert list(target.shape) == [1, 4, 3]
    assert target.dtype == torch.float32
