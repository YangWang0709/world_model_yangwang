from pathlib import Path

import numpy as np

from data.bridgedata_v2_tfds_clip_cache import (
    clip_cache_to_torch_videos,
    load_clip_cache_npz,
    validate_clip_cache,
)


def _write_npz(path: Path, channel_last: bool) -> None:
    if channel_last:
        context = np.zeros((16, 224, 224, 3), dtype=np.uint8)
        current = np.zeros((4, 224, 224, 3), dtype=np.uint8)
        future = np.zeros((4, 224, 224, 3), dtype=np.uint8)
    else:
        context = np.zeros((16, 3, 224, 224), dtype=np.uint8)
        current = np.zeros((4, 3, 224, 224), dtype=np.uint8)
        future = np.zeros((4, 3, 224, 224), dtype=np.uint8)
    np.savez_compressed(
        path,
        context_video=context,
        current_video=current,
        future_video=future,
        metadata_json=np.array(
            '{"action_used_as_input": false, "language_used_as_input": false, "goal_used_as_input": false}'
        ),
    )


def test_clip_cache_loads_channel_last_and_channel_first(tmp_path: Path):
    for channel_last in (True, False):
        path = tmp_path / f"sample_{channel_last}.npz"
        _write_npz(path, channel_last=channel_last)
        sample = load_clip_cache_npz(path)
        assert validate_clip_cache(sample)
        videos = clip_cache_to_torch_videos(sample)
        assert list(videos["context_video"].shape) == [16, 3, 224, 224]
        assert list(videos["current_video"].shape) == [4, 3, 224, 224]
        assert list(videos["future_video"].shape) == [4, 3, 224, 224]
        assert videos["context_video"].dtype.is_floating_point
        assert videos["metadata"]["action_used_as_input"] is False
