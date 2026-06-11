"""Encoder wrappers for the minimal TGP-AWB skeleton."""

from .dummy_video_encoder import DummyVideoEncoder
from .frozen_video_encoder import FrozenVideoEncoder, build_frozen_video_encoder

__all__ = ["DummyVideoEncoder", "FrozenVideoEncoder", "build_frozen_video_encoder"]
