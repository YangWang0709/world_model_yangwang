"""Unified frozen video encoder interface for Step 9B."""

from __future__ import annotations

from typing import Any

import torch

from .dummy_video_encoder import DummyVideoEncoder
from .videomae_wrapper import VideoMAEWrapper
from .vjepa_wrapper import VJEPAWrapper


class DummyFrozenVideoEncoder:
    """Adapter that exposes DummyVideoEncoder through the frozen encoder API."""

    encoder_name = "dummy_video_encoder"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.device = self._resolve_device(self.config.get("device"))
        self.model = DummyVideoEncoder(
            num_tokens=int(self.config.get("num_tokens", self.config.get("dummy_num_tokens", 196))),
            token_dim=int(self.config.get("token_dim", self.config.get("dummy_token_dim", 768))),
        ).to(self.device)
        self.model.eval()
        self.availability = {
            "available": True,
            "reason": "dummy_video_encoder is always available",
        }

    @staticmethod
    def _resolve_device(device_name: str | None) -> torch.device:
        if device_name is None or device_name == "cuda_if_available":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

    def is_available(self) -> bool:
        return True

    def encode(self, video_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(video_batch.to(self.device)).detach().cpu()


class FrozenVideoEncoder:
    """Dispatch wrapper with one encode API for dummy, VideoMAE, and V-JEPA."""

    def __init__(self, encoder_name: str, config: dict[str, Any] | None = None) -> None:
        self.requested_encoder_name = encoder_name
        self.config = dict(config or {})
        if encoder_name == "dummy_video_encoder":
            self.impl = DummyFrozenVideoEncoder(self.config)
        elif encoder_name == "videomae":
            self.impl = VideoMAEWrapper(self.config)
        elif encoder_name == "vjepa":
            self.impl = VJEPAWrapper(self.config)
        else:
            raise ValueError(
                f"Unsupported frozen video encoder {encoder_name!r}; "
                "expected one of: dummy_video_encoder, videomae, vjepa"
            )
        self.encoder_name = getattr(self.impl, "encoder_name", encoder_name)

    @property
    def availability(self) -> dict[str, Any]:
        return dict(getattr(self.impl, "availability", {"available": self.is_available()}))

    def is_available(self) -> bool:
        return bool(self.impl.is_available())

    def encode(self, video_batch: torch.Tensor) -> torch.Tensor:
        if video_batch.ndim != 5:
            raise ValueError(f"Expected video_batch shape [B, T, C, H, W], got {tuple(video_batch.shape)}")
        if not torch.is_floating_point(video_batch):
            raise TypeError("video_batch must be a floating point tensor")
        if not self.is_available():
            reason = self.availability.get("reason", "encoder unavailable")
            raise RuntimeError(f"Frozen encoder {self.requested_encoder_name!r} is unavailable: {reason}")
        tokens = self.impl.encode(video_batch)
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("Frozen encoder output must be a torch.Tensor")
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(1)
        if tokens.ndim != 3:
            raise ValueError(f"Frozen encoder output must have shape [B, N, D], got {tuple(tokens.shape)}")
        return tokens.contiguous()


def build_frozen_video_encoder(config: dict[str, Any]) -> FrozenVideoEncoder:
    """Build a frozen video encoder without forcing heavy optional imports at module import time."""

    encoder_name = str(config.get("name", "dummy_video_encoder"))
    return FrozenVideoEncoder(encoder_name=encoder_name, config=config)
