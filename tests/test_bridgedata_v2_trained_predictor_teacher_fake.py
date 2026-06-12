import torch

from models.bridgedata_v2_trained_predictor_teacher import (
    CurrentConditionedContextAttentionPredictor,
    future_token_summary,
)


def test_trained_predictor_teacher_forward_shapes_and_attention_sum():
    torch.manual_seed(0)
    model = CurrentConditionedContextAttentionPredictor(hidden_dim=32, seed=0)
    context = torch.randn(2, 16, 392, 768)
    current = torch.randn(2, 4, 392, 768)
    pred, attention = model(context, current, return_attention=True)
    assert list(pred.shape) == [2, 768]
    assert list(attention.shape) == [2, 6272]
    assert torch.allclose(attention.sum(dim=1), torch.ones(2), atol=1e-5)
    out = model.forward_from_summaries(torch.randn(2, 768), torch.randn(2, 768))
    assert list(out.shape) == [2, 768]


def test_future_summary_shape_and_no_action_language_inputs():
    future = torch.randn(2, 4, 392, 768)
    summary = future_token_summary(future)
    assert list(summary.shape) == [2, 768]
