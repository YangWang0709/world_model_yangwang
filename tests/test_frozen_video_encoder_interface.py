"""Tests for the Step 9B frozen video encoder interface."""

from __future__ import annotations

import pytest
import torch

from encoders.frozen_video_encoder import build_frozen_video_encoder


def test_build_dummy_frozen_encoder_and_encode() -> None:
    encoder = build_frozen_video_encoder(
        {
            "name": "dummy_video_encoder",
            "num_tokens": 5,
            "token_dim": 7,
            "device": "cpu",
        }
    )
    video = torch.rand(2, 4, 3, 16, 16)
    tokens = encoder.encode(video)

    assert encoder.is_available()
    assert list(tokens.shape) == [2, 5, 7]
    assert torch.isfinite(tokens).all()


def test_unknown_frozen_encoder_has_clear_error() -> None:
    with pytest.raises(ValueError, match="Unsupported frozen video encoder"):
        build_frozen_video_encoder({"name": "unknown_encoder"})
