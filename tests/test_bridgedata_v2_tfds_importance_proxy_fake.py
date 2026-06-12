import torch

from data.bridgedata_v2_tfds_importance_proxy import METHOD, compute_context_predictive_importance


def _fake_tokens():
    generator = torch.Generator().manual_seed(123)
    return {
        "context": torch.randn((16, 392, 768), generator=generator),
        "current": torch.randn((4, 392, 768), generator=generator),
        "future": torch.randn((4, 392, 768), generator=generator),
    }


def test_proxy_generates_context_only_importance_shapes_and_norms():
    tokens = _fake_tokens()
    result = compute_context_predictive_importance(tokens["context"], tokens["current"], tokens["future"])
    assert result["method"] == METHOD
    assert list(result["context_importance_raw"].shape) == [16, 392]
    assert list(result["context_importance_norm"].shape) == [16, 392]
    assert list(result["temporal_importance"].shape) == [16]
    assert list(result["spatial_importance"].shape) == [392]
    assert float(result["context_importance_norm"].min()) >= 0.0
    assert float(result["context_importance_norm"].max()) <= 1.0
    assert result["stats"]["current_tokens_kept_full"] is True
    assert result["stats"]["train_current_importance"] is False
    assert "current_importance" not in result
    assert result["context_importance_norm"].requires_grad is False


def test_proxy_is_deterministic():
    tokens = _fake_tokens()
    first = compute_context_predictive_importance(tokens["context"], tokens["current"], tokens["future"])
    second = compute_context_predictive_importance(tokens["context"], tokens["current"], tokens["future"])
    assert torch.equal(first["context_importance_raw"], second["context_importance_raw"])
    assert torch.equal(first["context_importance_norm"], second["context_importance_norm"])
