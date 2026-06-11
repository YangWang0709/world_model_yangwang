"""Data utilities for the minimal TGP-AWB world model skeleton."""

__all__ = [
    "DummyVideoDataset",
    "RealVideoClipDataset",
    "TokenShardDataset",
    "VideoClipDataset",
    "token_shard_collate_fn",
]


def __getattr__(name: str):
    if name == "DummyVideoDataset":
        from .datasets import DummyVideoDataset

        return DummyVideoDataset
    if name == "RealVideoClipDataset":
        from .real_video_dataset import RealVideoClipDataset

        return RealVideoClipDataset
    if name in {"TokenShardDataset", "token_shard_collate_fn"}:
        from .token_shard_dataset import TokenShardDataset, token_shard_collate_fn

        return TokenShardDataset if name == "TokenShardDataset" else token_shard_collate_fn
    if name == "VideoClipDataset":
        from .video_clip_dataset import VideoClipDataset

        return VideoClipDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
