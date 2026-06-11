from __future__ import annotations

import torch

from models.selection_policies import (
    LearnedSelectorPolicy,
    RandomKPolicy,
    TeacherImportanceTopKPolicy,
    UniformKPolicy,
    compute_teacher_importance_selection_metrics,
)


def test_bair_random_k_policy_outputs_unique_indices() -> None:
    tokens = torch.randn(3, 8, 4)
    result = RandomKPolicy(seed=7).select(tokens, k=3)

    assert result["selected_tokens"].shape == (3, 3, 4)
    assert result["selected_indices"].shape == (3, 3)
    for row in result["selected_indices"]:
        assert len(set(row.tolist())) == 3


def test_bair_uniform_k_policy_outputs_stable_indices() -> None:
    tokens = torch.randn(2, 10, 4)
    result = UniformKPolicy().select(tokens, k=4)

    assert result["selected_indices"].tolist() == [[0, 3, 6, 9], [0, 3, 6, 9]]
    assert result["selected_tokens"].shape == (2, 4, 4)


def test_bair_teacher_importance_topk_policy_selects_target_topk() -> None:
    tokens = torch.arange(1 * 5 * 1, dtype=torch.float32).reshape(1, 5, 1)
    batch = {"importance_scores_norm": torch.tensor([[0.1, 0.9, 0.2, 1.0, 0.0]])}

    result = TeacherImportanceTopKPolicy().select(tokens, batch=batch, k=2)

    assert result["selected_indices"].tolist() == [[3, 1]]
    assert result["selected_tokens"].squeeze(-1).tolist() == [[3.0, 1.0]]


class ScoreByFirstDim(torch.nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens[..., 0]


def test_bair_learned_selector_policy_accepts_mock_selector() -> None:
    tokens = torch.tensor(
        [
            [[0.1, 0.0], [2.0, 0.0], [1.0, 0.0]],
            [[3.0, 0.0], [0.2, 0.0], [2.5, 0.0]],
        ]
    )
    result = LearnedSelectorPolicy(selector=ScoreByFirstDim()).select(tokens, k=2)

    assert result["selected_indices"].tolist() == [[1, 2], [0, 2]]
    assert result["scores"].shape == (2, 3)


def test_bair_teacher_importance_selection_metrics() -> None:
    selected_indices = torch.tensor([[3, 1], [0, 2]])
    importance = torch.tensor([[0.1, 0.9, 0.2, 1.0], [1.0, 0.0, 0.8, 0.1]])

    metrics = compute_teacher_importance_selection_metrics(selected_indices, importance, k=2, num_tokens=4)

    assert metrics["selector_target_top1_overlap"] == 1.0
    assert metrics["selector_target_topk_overlap"] == 1.0
    assert metrics["selected_teacher_importance_mean"] > metrics["random_teacher_importance_mean"]
    assert metrics["selected_vs_random_importance_gap"] > 0.0
    assert metrics["token_retention_ratio"] == 0.5
