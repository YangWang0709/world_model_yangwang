from data.bridgedata_v2_rlds_field_resolver import resolve_rlds_fields


def test_image_resolver_prefers_real_observation_image_over_metadata_flags():
    resolved = resolve_rlds_fields(
        {
            "image_fields": [
                "episode_metadata/has_image_0",
                "episode_metadata/has_image_1",
                "steps/observation/image_1",
                "steps/observation/image_0",
            ],
            "action_fields": ["steps/action"],
            "language_fields": ["steps/language_embedding", "steps/language_instruction"],
            "goal_fields": [],
        }
    )
    assert resolved["image_field"] == "steps/observation/image_0"
    assert resolved["image_field_valid"] is True
    assert resolved["image_field_is_metadata_flag"] is False
    assert any(item["field"] == "episode_metadata/has_image_0" for item in resolved["image_rejected_fields"])
    assert resolved["action_field"] == "steps/action"
    assert resolved["action_used_as_input"] is False
    assert resolved["language_field"] == "steps/language_instruction"
    assert resolved["language_is_text"] is True
    assert resolved["language_used_as_input"] is False
    assert resolved["goal_field"] is None
    assert resolved["goal_used_as_input"] is False


def test_only_metadata_image_flags_blocks_manifest_resolution():
    resolved = resolve_rlds_fields(
        {
            "image_fields": ["episode_metadata/has_image_0"],
            "action_fields": ["steps/action"],
            "language_fields": ["steps/language_instruction"],
            "goal_fields": [],
        }
    )
    assert resolved["image_field"] is None
    assert resolved["image_field_valid"] is False
    assert resolved["blocking_errors"]


def test_language_embedding_is_last_resort_with_warning():
    resolved = resolve_rlds_fields(
        {
            "image_fields": ["steps/observation/image_0"],
            "action_fields": ["steps/action"],
            "language_fields": ["steps/language_embedding"],
            "goal_fields": [],
        }
    )
    assert resolved["language_field"] == "steps/language_embedding"
    assert resolved["language_is_text"] is False
    assert resolved["warnings"]
