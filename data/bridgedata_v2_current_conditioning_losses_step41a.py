"""Loss helpers for Step41A current-conditioning diagnosis."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def current_conditioning_mse_loss(
    model_output: dict[str, Any] | torch.Tensor,
    target: torch.Tensor,
    config: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    logits = model_output["scores"] if isinstance(model_output, dict) else model_output
    if logits.shape != target.shape or logits.ndim != 3:
        raise ValueError(f"logits and target must be [B,T,S], got {tuple(logits.shape)} and {tuple(target.shape)}")
    use_sigmoid = bool((config or {}).get("use_sigmoid_scores", True))
    pred = torch.sigmoid(logits) if use_sigmoid else logits
    mse = F.mse_loss(pred, target, reduction="mean")
    return {"loss": mse, "mse_loss": mse.detach()}
