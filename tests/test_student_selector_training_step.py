from __future__ import annotations

import torch

from models.attention_selector import AttentionSelector
from training.losses import importance_regression_loss, ranking_margin_loss


def test_student_selector_training_step_backward_and_optimizer() -> None:
    model = AttentionSelector(token_dim=6, hidden_dim=8, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randn(2, 5, 6)
    targets = torch.rand(2, 5)
    key_mask = torch.tensor(
        [
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
        ],
        dtype=torch.float32,
    )

    logits = model(tokens)
    scores = torch.sigmoid(logits)
    loss = importance_regression_loss(scores, targets) + 0.1 * ranking_margin_loss(scores, key_mask)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert logits.shape == (2, 5)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(loss)
