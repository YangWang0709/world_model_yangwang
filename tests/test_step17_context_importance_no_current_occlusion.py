from pathlib import Path

import torch
from torch import nn

from scripts.generate_context_predictive_importance import context_importance_for_batch


class RecordingTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_calls = []
        self.current_calls = []

    def forward(self, context_tokens: torch.Tensor, current_tokens: torch.Tensor) -> torch.Tensor:
        self.context_calls.append(context_tokens.detach().clone())
        self.current_calls.append(current_tokens.detach().clone())
        return context_tokens.mean(dim=1) + current_tokens.mean(dim=1)


def test_context_importance_masks_only_context_and_keeps_current_full():
    teacher = RecordingTeacher()
    context = torch.randn(2, 4, 6)
    current = torch.randn(2, 3, 6)
    future = torch.randn(2, 2, 6)
    original_current = current.clone()
    values = context_importance_for_batch(
        teacher,
        context,
        current,
        future,
        token_chunk_size=2,
        mask_value=0.0,
    )
    assert values["importance_scores"].shape == (2, 4)
    assert torch.equal(current, original_current)
    assert len(teacher.current_calls) == 1 + context.shape[1]
    for seen_current in teacher.current_calls:
        assert torch.equal(seen_current, original_current)
    base_context = teacher.context_calls[0]
    assert torch.equal(base_context, context)
    for idx, masked_context in enumerate(teacher.context_calls[1:]):
        changed = (masked_context != context).any(dim=(0, 2)).nonzero(as_tuple=False).flatten().tolist()
        assert changed == [idx]
        assert torch.all(masked_context[:, idx, :] == 0.0)


def test_context_importance_metadata_contract_is_static():
    source = Path("scripts/generate_context_predictive_importance.py").read_text(encoding="utf-8")
    assert '"occlude_only": "context_tokens"' in source
    assert '"keep_current_full": True' in source
    assert '"current_tokens_unchanged": True' in source
    assert '"trained_current_importance": False' in source
    assert "current_importance_scores" not in source

