import inspect

import pytest
import torch
from torch import nn

from models.context_bottleneck_world_model import ContextBottleneckWorldModel
from training.train_context_bottleneck_world_model import save_context_bottleneck_checkpoint


class CaptureHead(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.last_input = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.last_input = value.detach().clone()
        return value[:, : self.output_dim]


def test_context_bottleneck_uses_full_current_mean_pool():
    model = ContextBottleneckWorldModel(token_dim=8, hidden_dim=16, output_dim=8)
    head = CaptureHead(output_dim=8)
    model.head = head
    current = torch.randn(2, 392, 8)
    selected_context = torch.randn(2, 32, 8)
    out = model(current, selected_context)
    assert out.shape == (2, 8)
    assert head.last_input is not None
    assert torch.allclose(head.last_input[:, 8:], current.mean(dim=1))
    shorter_current = current[:, :128].contiguous()
    model(shorter_current, selected_context)
    assert torch.allclose(head.last_input[:, 8:], shorter_current.mean(dim=1))


def test_context_bottleneck_rejects_bad_current_shape():
    model = ContextBottleneckWorldModel(token_dim=8, hidden_dim=16, output_dim=8)
    with pytest.raises(ValueError):
        model(torch.randn(2, 392), torch.randn(2, 32, 8))


def test_context_bottleneck_forward_has_no_current_topk_or_gather():
    source = inspect.getsource(ContextBottleneckWorldModel.forward)
    assert "current_tokens.mean(dim=1)" in source
    assert "topk" not in source.lower()
    assert "gather" not in source.lower()


def test_context_bottleneck_checkpoint_records_no_current_drop(tmp_path):
    model = ContextBottleneckWorldModel(token_dim=8, hidden_dim=16, output_dim=8)
    summary = {"current_tokens_dropped": False}
    path = save_context_bottleneck_checkpoint(tmp_path / "model.pt", model, None, 1, {"token_dim": 8}, summary)
    payload = torch.load(path, map_location="cpu")
    assert payload["current_tokens_dropped"] is False
    assert payload["metrics_summary"]["current_tokens_dropped"] is False

