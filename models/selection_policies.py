"""Token selection policies used for Step 8 baseline comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.attention_selector import AttentionSelector
from training.student_selector_trainer import load_student_selector_checkpoint


def gather_tokens_by_indices(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather `[B, K, D]` tokens from `[B, N, D]` by per-sample indices."""

    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [B, N, D], got {tuple(tokens.shape)}")
    if indices.ndim != 2:
        raise ValueError(f"indices must be [B, K], got {tuple(indices.shape)}")
    if tokens.shape[0] != indices.shape[0]:
        raise ValueError("tokens and indices must have the same batch size")
    if indices.numel() and (int(indices.min().item()) < 0 or int(indices.max().item()) >= tokens.shape[1]):
        raise IndexError("selection indices are out of token range")
    gather_index = indices.to(tokens.device).long().unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    return torch.gather(tokens, dim=1, index=gather_index).contiguous()


def compute_selection_metrics(
    selected_indices: torch.Tensor,
    key_token_mask: torch.Tensor,
    num_tokens: int,
) -> dict[str, float]:
    """Compute key-token recovery metrics for selected token indices."""

    if selected_indices.ndim != 2:
        raise ValueError(f"selected_indices must be [B, K], got {tuple(selected_indices.shape)}")
    if key_token_mask.ndim != 2:
        raise ValueError(f"key_token_mask must be [B, N], got {tuple(key_token_mask.shape)}")
    if selected_indices.shape[0] != key_token_mask.shape[0]:
        raise ValueError("selected_indices and key_token_mask batch sizes must match")
    if key_token_mask.shape[1] != num_tokens:
        raise ValueError(f"num_tokens={num_tokens} does not match mask width={key_token_mask.shape[1]}")
    if selected_indices.numel() and (
        int(selected_indices.min().item()) < 0 or int(selected_indices.max().item()) >= num_tokens
    ):
        raise IndexError("selected_indices are out of token range")

    selected_indices = selected_indices.to(key_token_mask.device).long()
    selected_mask = torch.gather(key_token_mask.float(), dim=1, index=selected_indices)
    top1_hits = []
    coverage_values = []
    key_fractions = []
    k = int(selected_indices.shape[1])
    for sample_selected_mask, sample_key_mask in zip(selected_mask, key_token_mask):
        key_count = int(sample_key_mask.float().sum().item())
        if key_count == 0:
            continue
        selected_key_count = float(sample_selected_mask.sum().item())
        top1_hits.append(float(sample_selected_mask[0].item()))
        coverage_values.append(selected_key_count / float(key_count))
        key_fractions.append(selected_key_count / float(max(k, 1)))

    selected_topk_hit_rate = float(sum(coverage_values) / len(coverage_values)) if coverage_values else 0.0
    return {
        "selected_top1_hit_rate": float(sum(top1_hits) / len(top1_hits)) if top1_hits else 0.0,
        "selected_topk_hit_rate": selected_topk_hit_rate,
        "selected_key_coverage": selected_topk_hit_rate,
        "selected_key_fraction": float(sum(key_fractions) / len(key_fractions)) if key_fractions else 0.0,
        "token_retention_ratio": float(k) / float(num_tokens),
    }


class BaseSelectionPolicy:
    """Base class for swappable Student token selection policies."""

    policy_name = "base"

    def select(
        self,
        past_tokens: torch.Tensor,
        batch: dict[str, Any] | None = None,
        k: int = 4,
        device: torch.device | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class RandomKPolicy(BaseSelectionPolicy):
    """Randomly select K unique tokens independently for each sample."""

    policy_name = "random_k"

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)

    def select(
        self,
        past_tokens: torch.Tensor,
        batch: dict[str, Any] | None = None,
        k: int = 4,
        device: torch.device | None = None,
    ) -> dict[str, Any]:
        del batch
        if k <= 0 or k > past_tokens.shape[1]:
            raise ValueError(f"k must be in [1, {past_tokens.shape[1]}], got {k}")
        indices = [torch.randperm(past_tokens.shape[1], generator=self.generator)[:k] for _ in range(past_tokens.shape[0])]
        selected_indices = torch.stack(indices, dim=0).to(device or past_tokens.device)
        return {
            "selected_tokens": gather_tokens_by_indices(past_tokens, selected_indices),
            "selected_indices": selected_indices,
            "scores": None,
            "policy_name": self.policy_name,
        }


class UniformKPolicy(BaseSelectionPolicy):
    """Select K approximately evenly spaced token indices."""

    policy_name = "uniform_k"

    def select(
        self,
        past_tokens: torch.Tensor,
        batch: dict[str, Any] | None = None,
        k: int = 4,
        device: torch.device | None = None,
    ) -> dict[str, Any]:
        del batch
        if k <= 0 or k > past_tokens.shape[1]:
            raise ValueError(f"k must be in [1, {past_tokens.shape[1]}], got {k}")
        indices_1d = torch.linspace(0, past_tokens.shape[1] - 1, steps=k).round().long()
        selected_indices = indices_1d.unsqueeze(0).expand(past_tokens.shape[0], -1).to(device or past_tokens.device)
        return {
            "selected_tokens": gather_tokens_by_indices(past_tokens, selected_indices),
            "selected_indices": selected_indices,
            "scores": None,
            "policy_name": self.policy_name,
        }


class OracleKeyPolicy(BaseSelectionPolicy):
    """Upper-bound baseline that selects known structured-toy key tokens first."""

    policy_name = "oracle_key"

    def select(
        self,
        past_tokens: torch.Tensor,
        batch: dict[str, Any] | None = None,
        k: int = 4,
        device: torch.device | None = None,
    ) -> dict[str, Any]:
        if batch is None or "key_token_mask" not in batch:
            raise KeyError("OracleKeyPolicy requires batch['key_token_mask']")
        if k <= 0 or k > past_tokens.shape[1]:
            raise ValueError(f"k must be in [1, {past_tokens.shape[1]}], got {k}")
        key_mask = batch["key_token_mask"].to("cpu").bool()
        selected = []
        for sample_mask in key_mask:
            key_indices = torch.nonzero(sample_mask, as_tuple=False).flatten().long()
            non_key_indices = torch.nonzero(~sample_mask, as_tuple=False).flatten().long()
            sample_indices = torch.cat([key_indices, non_key_indices], dim=0)[:k]
            if sample_indices.numel() < k:
                raise ValueError("Not enough token indices to fill oracle selection")
            selected.append(sample_indices)
        selected_indices = torch.stack(selected, dim=0).to(device or past_tokens.device)
        return {
            "selected_tokens": gather_tokens_by_indices(past_tokens, selected_indices),
            "selected_indices": selected_indices,
            "scores": None,
            "policy_name": self.policy_name,
        }


class LearnedSelectorPolicy(BaseSelectionPolicy):
    """Use the Step 6 AttentionSelector checkpoint as a frozen top-k policy."""

    policy_name = "learned_selector"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        selector: torch.nn.Module | None = None,
        map_location: str | torch.device = "cpu",
    ) -> None:
        if selector is None and checkpoint_path is None:
            raise ValueError("LearnedSelectorPolicy requires a selector or checkpoint_path")
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else ""
        if selector is not None:
            self.selector = selector
        else:
            checkpoint = load_student_selector_checkpoint(checkpoint_path, map_location=map_location)
            self.selector = AttentionSelector(**checkpoint["model_config"])
            self.selector.load_state_dict(checkpoint["model_state_dict"])
        self.selector.eval()
        for parameter in self.selector.parameters():
            parameter.requires_grad = False

    def select(
        self,
        past_tokens: torch.Tensor,
        batch: dict[str, Any] | None = None,
        k: int = 4,
        device: torch.device | None = None,
    ) -> dict[str, Any]:
        del batch
        if k <= 0 or k > past_tokens.shape[1]:
            raise ValueError(f"k must be in [1, {past_tokens.shape[1]}], got {k}")
        target_device = device or past_tokens.device
        self.selector.to(target_device)
        self.selector.eval()
        with torch.no_grad():
            logits = self.selector(past_tokens.to(target_device))
            scores = torch.sigmoid(logits)
            selected_indices = torch.topk(scores, k=k, dim=1).indices
        return {
            "selected_tokens": gather_tokens_by_indices(past_tokens, selected_indices),
            "selected_indices": selected_indices,
            "scores": scores,
            "policy_name": self.policy_name,
        }


def build_selection_policy(
    policy_name: str,
    seed: int = 0,
    learned_selector_checkpoint: str | Path | None = None,
) -> BaseSelectionPolicy:
    if policy_name == "random_k":
        return RandomKPolicy(seed=seed)
    if policy_name == "uniform_k":
        return UniformKPolicy()
    if policy_name == "oracle_key":
        return OracleKeyPolicy()
    if policy_name == "learned_selector":
        return LearnedSelectorPolicy(checkpoint_path=learned_selector_checkpoint)
    raise ValueError(f"Unknown selection policy: {policy_name!r}")
