import torch

from models.context_bottleneck_world_model import ContextBottleneckWorldModel
from models.context_selection_policies import select_context_tokens


def test_context_bottleneck_uses_full_current_and_selected_context_backward():
    model = ContextBottleneckWorldModel(token_dim=768, hidden_dim=128, output_dim=768)
    current = torch.randn(2, 392, 768)
    context = torch.randn(2, 784, 768)
    selection = select_context_tokens(
        context_tokens=context,
        current_tokens=current,
        policy="uniform_context_topK",
        topk=32,
    )
    pred = model(current, selection["selected_tokens"])
    assert pred.shape == (2, 768)
    loss = pred.pow(2).mean()
    loss.backward()
    assert selection["selected_tokens"].shape == (2, 32, 768)
    assert current.shape == (2, 392, 768)
    assert any(param.grad is not None for param in model.parameters())
