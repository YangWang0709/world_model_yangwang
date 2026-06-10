from __future__ import annotations

import torch

from models.selection_policies import (
    LearnedSelectorPolicy,
    OracleKeyPolicy,
    RandomKPolicy,
    UniformKPolicy,
    compute_selection_metrics,
    gather_tokens_by_indices,
)


def test_random_k_policy_outputs_unique_indices() -> None:
    tokens = torch.randn(3, 8, 4)
    result = RandomKPolicy(seed=7).select(tokens, k=3)

    assert result["selected_tokens"].shape == (3, 3, 4)
    assert result["selected_indices"].shape == (3, 3)
    for row in result["selected_indices"]:
        assert len(set(row.tolist())) == 3


def test_uniform_k_policy_uses_same_even_indices_for_all_samples() -> None:
    tokens = torch.randn(2, 10, 4)
    result = UniformKPolicy().select(tokens, k=4)

    assert result["selected_indices"].tolist() == [[0, 3, 6, 9], [0, 3, 6, 9]]
    assert result["selected_tokens"].shape == (2, 4, 4)


def test_oracle_key_policy_prefers_key_tokens_then_fills() -> None:
    tokens = torch.arange(2 * 6 * 1, dtype=torch.float32).reshape(2, 6, 1)
    batch = {
        "key_token_mask": torch.tensor(
            [
                [0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
            ],
            dtype=torch.float32,
        )
    }
    result = OracleKeyPolicy().select(tokens, batch=batch, k=3)

    assert result["selected_indices"].tolist() == [[1, 3, 0], [4, 0, 1]]
    assert result["selected_tokens"].shape == (2, 3, 1)


class ScoreByFirstDim(torch.nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens[..., 0]


def test_learned_selector_policy_accepts_mock_selector() -> None:
    tokens = torch.tensor(
        [
            [[0.1, 0.0], [2.0, 0.0], [1.0, 0.0]],
            [[3.0, 0.0], [0.2, 0.0], [2.5, 0.0]],
        ]
    )
    policy = LearnedSelectorPolicy(selector=ScoreByFirstDim())
    result = policy.select(tokens, k=2)

    assert result["selected_indices"].tolist() == [[1, 2], [0, 2]]
    assert result["scores"].shape == (2, 3)


def test_gather_tokens_by_indices_and_selection_metrics() -> None:
    tokens = torch.arange(1 * 5 * 2, dtype=torch.float32).reshape(1, 5, 2)
    indices = torch.tensor([[3, 1]])
    gathered = gather_tokens_by_indices(tokens, indices)
    key_mask = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.float32)
    metrics = compute_selection_metrics(indices, key_mask, num_tokens=5)

    assert gathered.tolist() == [[[6.0, 7.0], [2.0, 3.0]]]
    assert metrics["selected_top1_hit_rate"] == 1.0
    assert metrics["selected_topk_hit_rate"] == 1.0
    assert metrics["selected_key_coverage"] == 1.0
    assert metrics["token_retention_ratio"] == 0.4
