from data.bridgedata_v2_rlds_schema_inspector import inspect_rlds_like_episodes


def _episode(length: int):
    return {
        "steps": [
            {
                "observation": {
                    "image_0": {"__array__": True, "shape": [224, 224, 3], "dtype": "uint8"},
                    "state": {"__array__": True, "shape": [7], "dtype": "float32"},
                    "goal_image": {"__array__": True, "shape": [224, 224, 3], "dtype": "uint8"},
                },
                "action": {"__array__": True, "shape": [7], "dtype": "float32"},
                "language_instruction": "put object in bowl",
            }
            for _ in range(length)
        ]
    }


def test_fake_rlds_schema_detects_candidate_fields_and_window_feasibility():
    summary = inspect_rlds_like_episodes([_episode(30), _episode(10)], min_trajectory_len=24)
    fields = summary["candidate_fields"]
    assert summary["num_episodes_scanned"] == 2
    assert summary["can_build_16_4_4_windows"] is True
    assert any("image_0" in item for item in fields["image_fields"])
    assert any("action" in item for item in fields["action_fields"])
    assert any("language_instruction" in item for item in fields["language_fields"])
    assert any("goal_image" in item for item in fields["goal_fields"])
    assert any("state" in item for item in fields["proprio_fields"])
    assert summary["save_video_tensors"] is False


def test_short_episode_cannot_build_16_4_4_windows():
    summary = inspect_rlds_like_episodes([_episode(10)], min_trajectory_len=24)
    assert summary["can_build_16_4_4_windows"] is False
