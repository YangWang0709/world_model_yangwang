import torch

from models.bridgedata_v2_current_conditioned_selector_step41a import VARIANT_NAMES, build_current_conditioned_selector


def test_step41a_current_conditioning_variants_emit_scores_and_detach_inputs():
    config = _config()
    context = torch.randn(2, 4, 5, 6, requires_grad=True)
    current = torch.randn(2, 2, 5, 6, requires_grad=True)

    for variant in VARIANT_NAMES:
        model = build_current_conditioned_selector(config, variant)
        output = model(context, current)

        assert output["variant_name"] == variant
        assert list(output["scores"].shape) == [2, 4, 5]
        assert torch.isfinite(output["scores"]).all()
        output["scores"].sum().backward()
        assert context.grad is None
        assert current.grad is None


def test_step41a_no_current_variant_does_not_use_current_tokens():
    config = _config()
    context = torch.randn(2, 4, 5, 6)
    current_a = torch.randn(2, 2, 5, 6)
    current_b = torch.randn(2, 2, 5, 6) + 100.0
    model = build_current_conditioned_selector(config, "no_current_context_only")

    a = model(context, current_a)
    b = model(context, current_b)
    none = model(context, None)

    assert torch.allclose(a["scores"], b["scores"])
    assert torch.allclose(a["scores"], none["scores"])
    assert a["diagnostics"]["uses_current_tokens"] is False


def test_step41a_coarse_spatial_query_uses_chunked_token_index_attention():
    config = _config()
    model = build_current_conditioned_selector(config, "current_coarse_spatial_query_attention")
    output = model(torch.randn(2, 4, 5, 6), torch.randn(2, 2, 5, 6))

    assert output["diagnostics"]["token_index_coarse_heuristic"] is True
    assert output["diagnostics"]["max_attention_elements_seen"] <= 2 * 4 * (2 * 3)


def _config():
    return {
        "data": {"token_dim": 6, "context_frames": 4, "current_frames": 2, "spatial_tokens": 5},
        "model": {"token_dim": 6, "hidden_dim": 8, "context_frames": 4, "current_frames": 2, "spatial_tokens": 5},
        "current_conditioning_variants": {
            "attention_dim": 4,
            "coarse_current_bins": 3,
            "chunk_context_tokens": 4,
            "attention_temperature": 1.0,
            "detach_token_inputs": True,
            "use_temporal_embedding": True,
            "use_spatial_embedding": True,
        },
    }
