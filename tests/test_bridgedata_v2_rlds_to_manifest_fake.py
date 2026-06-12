from data.bridgedata_v2_manifest_schema import validate_bridgedata_manifest_record
from data.bridgedata_v2_rlds_to_manifest import fake_rlds_episodes_to_manifest_records
from data.bridgedata_v2_window_builder import build_bridgedata_windows_from_manifest
from data.long_context_window_spec import LongContextWindowSpec


def test_fake_rlds_episodes_convert_to_manifest_and_windows():
    episode = {
        "steps": [
            {
                "observation": {"image_0": {"__array__": True, "shape": [224, 224, 3], "dtype": "uint8"}},
                "action": {"__array__": True, "shape": [7], "dtype": "float32"},
                "language_instruction": "move object",
            }
            for _ in range(30)
        ]
    }
    records = fake_rlds_episodes_to_manifest_records([episode])
    assert len(records) == 1
    record = records[0]
    assert validate_bridgedata_manifest_record(record)
    assert record["frame_paths"] == []
    assert record["actions"] is None
    assert record["metadata"]["action_field"] is not None
    assert record["metadata"]["use_action_as_input"] is False
    windows = build_bridgedata_windows_from_manifest(records, LongContextWindowSpec(16, 4, 4))
    assert len(windows) == 7
    assert windows[0]["metadata"]["use_action_as_input"] is False
    assert windows[0]["context_video"] is None
