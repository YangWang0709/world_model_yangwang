"""Conservative VideoMAE wrapper for Step 9C.

The wrapper never downloads weights unless allow_download is explicitly enabled.
Heavy optional imports are intentionally kept inside methods.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


class VideoMAEWrapper:
    encoder_name = "videomae"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.model_name_or_path = self.config.get("model_name_or_path")
        self.allow_download = bool(self.config.get("allow_download", False))
        self.local_files_only = bool(self.config.get("local_files_only", not self.allow_download))
        if not self.allow_download:
            self.local_files_only = True
        self.cache_dir = self.config.get("cache_dir")
        self.image_size = int(self.config.get("image_size", 224))
        self.num_frames = int(self.config.get("num_frames", 8))
        self.output_mode = str(self.config.get("output_mode", "last_hidden_state"))
        self.device = self._resolve_device(self.config.get("device"))
        self.ignore_mismatched_sizes = bool(self.config.get("ignore_mismatched_sizes", True))
        self.model = None
        self.pretrained_num_frames: int | None = None
        self.effective_num_frames: int | None = None
        self.last_encode_summary: dict[str, Any] = {}
        self.availability = self._load_model_if_possible()

    @staticmethod
    def transformers_available() -> bool:
        return importlib.util.find_spec("transformers") is not None

    @staticmethod
    def _resolve_device(device_name: str | None) -> torch.device:
        if device_name is None or device_name == "cuda_if_available":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

    def _unavailable(self, reason: str) -> dict[str, Any]:
        return {
            "available": False,
            "reason": reason,
            "model_name_or_path": self.model_name_or_path,
            "cache_dir": self.cache_dir,
            "allow_download": self.allow_download,
            "local_files_only": self.local_files_only,
            "num_frames": self.num_frames,
            "image_size": self.image_size,
            "output_mode": self.output_mode,
        }

    def _load_model_if_possible(self) -> dict[str, Any]:
        if not self.transformers_available():
            return self._unavailable("transformers is not installed")
        if not self.model_name_or_path:
            return self._unavailable("model_name_or_path is not configured")
        if self.local_files_only and not Path(str(self.model_name_or_path)).exists():
            # A named Hugging Face model may still exist in cache, so we let transformers try it.
            pass

        try:
            from transformers import VideoMAEConfig, VideoMAEModel  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional package version
            return self._unavailable(f"transformers is present but VideoMAEModel import failed: {exc}")

        try:
            pretrained_config = VideoMAEConfig.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
            )
            self.pretrained_num_frames = int(getattr(pretrained_config, "num_frames", self.num_frames))
            pretrained_config.num_frames = self.num_frames
            pretrained_config.image_size = self.image_size
            model = VideoMAEModel.from_pretrained(
                self.model_name_or_path,
                config=pretrained_config,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
                ignore_mismatched_sizes=self.ignore_mismatched_sizes,
            )
        except Exception as exc:
            if not self.allow_download:
                return self._unavailable(
                    "VideoMAE weights are not available locally and allow_download=false; "
                    f"load error: {exc}"
                )
            return self._unavailable(f"VideoMAE model load failed: {exc}")

        model.eval()
        model.requires_grad_(False)
        self.model = model.to(self.device)
        self.effective_num_frames = int(getattr(model.config, "num_frames", self.num_frames))
        return {
            "available": True,
            "reason": "VideoMAE model loaded",
            "model_name_or_path": self.model_name_or_path,
            "cache_dir": self.cache_dir,
            "allow_download": self.allow_download,
            "local_files_only": self.local_files_only,
            "device": str(self.device),
            "pretrained_num_frames": self.pretrained_num_frames,
            "effective_num_frames": self.effective_num_frames,
            "image_size": self.image_size,
            "output_mode": self.output_mode,
            "ignore_mismatched_sizes": self.ignore_mismatched_sizes,
        }

    def _adapt_video_batch(self, video_batch: torch.Tensor) -> torch.Tensor:
        """Fit external conservative clips to the configured VideoMAE input shape."""

        adapted = video_batch
        batch_size, frames, channels, height, width = adapted.shape
        if frames != self.num_frames:
            if frames <= 0:
                raise ValueError("VideoMAE input must contain at least one frame")
            frame_indices = torch.linspace(0, frames - 1, self.num_frames, device=adapted.device).round().long()
            adapted = adapted.index_select(dim=1, index=frame_indices)

        if height != self.image_size or width != self.image_size:
            adapted = adapted.reshape(batch_size * self.num_frames, channels, height, width)
            adapted = F.interpolate(
                adapted,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            adapted = adapted.reshape(batch_size, self.num_frames, channels, self.image_size, self.image_size)
        return adapted

    def is_available(self) -> bool:
        return bool(self.availability.get("available", False))

    def encode(self, video_batch: torch.Tensor) -> torch.Tensor:
        if not self.is_available() or self.model is None:
            raise RuntimeError(f"VideoMAE unavailable: {self.availability.get('reason')}")
        if video_batch.ndim != 5:
            raise ValueError(f"Expected [B, T, C, H, W], got {tuple(video_batch.shape)}")
        input_shape = tuple(video_batch.shape)
        try:
            pixel_values = self._adapt_video_batch(video_batch).to(self.device)
            first_parameter = next(self.model.parameters(), None)
            if first_parameter is not None and torch.is_floating_point(pixel_values):
                pixel_values = pixel_values.to(dtype=first_parameter.dtype)
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - hardware dependent
            raise RuntimeError(
                "VideoMAE CUDA OOM with Step 9C conservative batch_size=1. "
                "Stop this stage and use a cloud 4090 or 48GB GPU server before retrying."
            ) from exc
        except RuntimeError as exc:
            message = str(exc)
            if "out of memory" in message.lower():
                raise RuntimeError(
                    "VideoMAE runtime OOM with Step 9C conservative batch_size=1. "
                    "Stop this stage and use a cloud 4090 or 48GB GPU server before retrying."
                ) from exc
            raise

        if self.output_mode == "pooler_output" and getattr(outputs, "pooler_output", None) is not None:
            tokens = outputs.pooler_output
        elif getattr(outputs, "last_hidden_state", None) is not None:
            tokens = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)) and outputs:
            tokens = outputs[0]
        else:
            raise RuntimeError("VideoMAE output did not include last_hidden_state or pooler_output")

        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(1)
        if tokens.ndim != 3:
            raise ValueError(f"VideoMAE tokens must have shape [B, N, D], got {tuple(tokens.shape)}")
        self.last_encode_summary = {
            "input_shape": list(input_shape),
            "effective_input_shape": list(pixel_values.shape),
            "output_shape": list(tokens.shape),
            "output_mode": self.output_mode,
            "device": str(self.device),
        }
        return tokens.detach().cpu().contiguous()
