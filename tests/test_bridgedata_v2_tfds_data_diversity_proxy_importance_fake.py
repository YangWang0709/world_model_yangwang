import torch

from data.bridgedata_v2_tfds_data_diversity_proxy_importance import fake_importance_summary


def test_fake_importance_summary_shape_and_topk_mass():
    values = torch.zeros(16, 392)
    values[0, 0] = 10.0
    values[1, 0] = 5.0
    summary = fake_importance_summary(values, topk=1)
    assert summary["context_importance_shape"] == [16, 392]
    assert summary["topk_mass"] == 10.0 / 15.0
    assert summary["current_importance_generated"] is False
    assert summary["temporal_concentration"] > 0.0


def test_fake_importance_requires_context_shape():
    try:
        fake_importance_summary(torch.zeros(4, 392), topk=1)
    except ValueError as exc:
        assert "[16, 392]" in str(exc)
    else:
        raise AssertionError("expected shape validation failure")
