import torch

from models.bridgedata_v2_proxy_factorized_selector import (
    ProxyFactorizedSelectorHead,
    ProxyFactorizedSelectorWithProxyTemporalPrior,
    proxy_patch_and_temporal_targets,
    validate_factorized_output,
)


def test_step37_factorized_selector_outputs_components_and_detaches_inputs():
    model = ProxyFactorizedSelectorHead(
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
    output = model(context, current)
    assert validate_factorized_output(output, batch=2, frames=4, spatial_tokens=3) is True
    assert list(output["scores"].shape) == [2, 4, 3]
    assert list(output["temporal_scores"].shape) == [2, 4]
    assert list(output["spatial_residual_scores"].shape) == [2, 4, 3]
    output["scores"].mean().backward()
    assert context.grad is None
    assert current.grad is None
    assert any(param.grad is not None for param in model.parameters())


def test_step37_proxy_temporal_prior_class_is_marker_only():
    model = ProxyFactorizedSelectorWithProxyTemporalPrior(token_dim=6, hidden_dim=8, context_frames=4, spatial_tokens=3)
    assert model.uses_proxy_temporal_auxiliary_target is True


def test_step37_targets_build_patch_and_temporal_labels():
    values = torch.rand(4, 3)
    targets = proxy_patch_and_temporal_targets(values)
    assert list(targets["proxy_patch_target"].shape) == [1, 4, 3]
    assert list(targets["proxy_temporal_target"].shape) == [1, 4]
