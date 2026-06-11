import torch

from models.context_bottleneck_world_model import ContextTeacherWorldModel, future_target_from_tokens


def test_context_teacher_forward_backward_shape():
    model = ContextTeacherWorldModel(token_dim=768, hidden_dim=128, output_dim=768)
    context = torch.randn(2, 784, 768)
    current = torch.randn(2, 392, 768)
    future = torch.randn(2, 392, 768)
    pred = model(context, current)
    assert pred.shape == (2, 768)
    target = future_target_from_tokens(future)
    loss = (pred - target).pow(2).mean()
    loss.backward()
    assert torch.isfinite(pred).all()
    assert any(param.grad is not None for param in model.parameters())
