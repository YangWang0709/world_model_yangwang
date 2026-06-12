"""Smoke-only BridgeData context bottleneck predictor for Step26."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class BridgeDataContextBottleneckSmokePredictor(torch.nn.Module):
    """A tiny random-init predictor used only for forward/loss plumbing checks."""

    def __init__(self, token_dim: int = 768, hidden_dim: int = 512, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.token_dim * 2, self.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.hidden_dim, self.token_dim),
        )
        self.random_init_result_not_scientific = True

    def forward(self, current_tokens: torch.Tensor, selected_context_tokens: torch.Tensor) -> torch.Tensor:
        current_summary = current_tokens.detach().to(dtype=torch.float32).mean(dim=(0, 1))
        if selected_context_tokens.numel() == 0:
            context_summary = torch.zeros_like(current_summary)
        else:
            context_summary = selected_context_tokens.detach().to(dtype=torch.float32).mean(dim=0)
        features = torch.cat([current_summary, context_summary], dim=0)
        return self.net(features)


def forward_loss_for_policy(
    sample: dict[str, Any],
    selection: dict[str, Any],
    model: torch.nn.Module,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        pred = model(sample["current_tokens"], selection["selected_context_tokens"])
        target = sample["future_tokens"].detach().to(dtype=torch.float32).mean(dim=(0, 1))
        loss = F.mse_loss(pred, target, reduction="mean")
    return {
        "sample_id": str(sample["sample_id"]),
        "trajectory_id": sample.get("trajectory_id"),
        "policy": str(selection["policy_name"]),
        "topk": selection.get("topk"),
        "num_selected": int(selection["num_selected"]),
        "loss": float(loss.item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "pred_shape": list(pred.shape),
        "target_shape": list(target.shape),
        "selected_importance_mass": float(selection["selected_importance_mass"]),
        "selected_importance_mean": float(selection["selected_importance_mean"]),
        "temporal_selected_histogram": list(selection["temporal_selected_histogram"]),
        "current_tokens_kept_full": True,
        "optimizer_step_performed": False,
        "training_performed": False,
        "random_init_result_not_scientific": True,
    }
