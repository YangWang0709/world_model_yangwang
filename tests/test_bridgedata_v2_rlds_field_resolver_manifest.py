from data.bridgedata_v2_rlds_to_manifest import schema_summary_to_manifest_records
from data.bridgedata_v2_window_builder import build_bridgedata_windows_from_manifest
from data.long_context_window_spec import LongContextWindowSpec


def _schema(image_fields):
    return {
        "candidate_fields": {
            "image_fields": image_fields,
            "action_fields": ["steps/action"],
            "language_fields": ["steps/language_embedding", "steps/language_instruction"],
            "goal_fields": [],
        },
        "episode_summaries": [
            {
                "episode_index": 0,
                "trajectory_id": "tfds_episode_000000",
                "num_steps": 30,
                "field_paths": image_fields,
            }
        ],
    }


def test_resolved_manifest_uses_real_image_field_not_metadata_flag():
    records = schema_summary_to_manifest_records(
        _schema(["episode_metadata/has_image_0", "steps/observation/image_0"]),
        min_frames=24,
        max_valid_trajectories=10,
    )
    assert len(records) == 1
    record = records[0]
    assert record["metadata"]["image_field"] == "steps/observation/image_0"
    assert record["metadata"]["image_field_is_metadata_flag"] is False
    assert "episode_metadata" not in record["metadata"]["image_field"]
    assert all("episode_metadata/has_image_" not in name for name in record["camera_names"])
    windows = build_bridgedata_windows_from_manifest(records, LongContextWindowSpec(16, 4, 4))
    assert len(windows) == 7


def test_invalid_image_field_generates_no_manifest_records():
    records = schema_summary_to_manifest_records(_schema(["episode_metadata/has_image_0"]))
    assert records == []
