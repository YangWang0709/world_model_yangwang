import json
from pathlib import Path

from data.bridgedata_v2_manifest_schema import load_bridgedata_manifest_jsonl
from data.bridgedata_v2_user_subset_manifest_tools import build_or_normalize_user_subset_manifest
from data.bridgedata_v2_user_subset_validator import validate_user_subset_manifest


def test_fake_realistic_subset_validates_without_reading_image_contents(tmp_path):
    subset = tmp_path / "subset"
    images = subset / "traj_000001" / "images"
    images.mkdir(parents=True)
    for idx in range(25):
        (images / f"frame_{idx:06d}.jpg").write_text("not a decoded image", encoding="utf-8")
    (subset / "traj_000001" / "metadata.json").write_text(
        json.dumps({"language_instruction": "pick up object", "camera_names": ["main"]}),
        encoding="utf-8",
    )
    (subset / "traj_000001" / "actions.npy").write_text("not loaded", encoding="utf-8")
    (subset / "traj_000001" / "goal.jpg").write_text("not opened", encoding="utf-8")
    manifest_path = tmp_path / "run" / "manifest.jsonl"
    build_or_normalize_user_subset_manifest(subset, manifest_path)
    records = load_bridgedata_manifest_jsonl(manifest_path)
    summary = validate_user_subset_manifest(records, subset, open_images=False)
    assert summary["real_format_validated"] is True
    assert summary["num_valid_trajectories"] == 1
    assert summary["checked_image_count"] == 25
    assert summary["opened_image_count"] == 0
    assert summary["use_action_as_input"] is False
    assert summary["use_language_as_input"] is False
    assert summary["use_goal_image_as_input"] is False
    assert summary["has_actions_likely"] is True
    assert summary["has_language_likely"] is True
    assert summary["has_goal_image_likely"] is True
