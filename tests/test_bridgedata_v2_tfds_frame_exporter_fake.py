from pathlib import Path

import numpy as np
import pytest

from data.bridgedata_v2_tfds_frame_exporter import export_window_to_clip_cache


def _steps(num_steps: int = 24):
    return [{"observation": {"image_0": np.full((8, 8, 3), index, dtype=np.uint8)}} for index in range(num_steps)]


def _window():
    return {
        "sample_id": "tfds_episode_000000_window_000000",
        "trajectory_id": "tfds_episode_000000",
        "split": "train",
        "context_frame_indices": list(range(16)),
        "current_frame_indices": list(range(16, 20)),
        "future_frame_indices": list(range(20, 24)),
        "metadata": {
            "record_metadata": {
                "tfds_episode_index": 0,
                "use_action_as_input": False,
                "use_language_as_input": False,
                "use_goal_image_as_input": False,
            }
        },
    }


def test_fake_rlds_episode_exports_npz_clip_cache_only(tmp_path: Path):
    record = export_window_to_clip_cache(
        window=_window(),
        steps=_steps(),
        resolved_fields={
            "image_field": "steps/observation/image_0",
            "image_field_valid": True,
            "action_field": "steps/action",
            "language_field": "steps/language_instruction",
            "goal_field": None,
        },
        clip_cache_dir=tmp_path,
        image_size=16,
    )
    assert record["context_shape"] == [16, 3, 16, 16]
    assert Path(record["clip_cache_path"]).suffix == ".npz"
    assert not list(tmp_path.glob("*.jpg"))
    assert not list(tmp_path.glob("*.png"))
    assert not list(tmp_path.glob("*.mp4"))


def test_fake_rlds_episode_refuses_metadata_image_flag(tmp_path: Path):
    with pytest.raises(ValueError, match="image_field"):
        export_window_to_clip_cache(
            window=_window(),
            steps=_steps(),
            resolved_fields={"image_field": "episode_metadata/has_image_0", "image_field_valid": True},
            clip_cache_dir=tmp_path,
            image_size=16,
        )
