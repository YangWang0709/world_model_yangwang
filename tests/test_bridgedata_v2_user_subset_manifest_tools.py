import json
from pathlib import Path

from data.bridgedata_v2_manifest_schema import load_bridgedata_manifest_jsonl
from data.bridgedata_v2_user_subset_manifest_tools import build_or_normalize_user_subset_manifest


def _make_traj(root: Path, name: str, frames: int = 30) -> None:
    images = root / name / "images"
    images.mkdir(parents=True)
    for idx in range(frames):
        (images / f"frame_{idx:06d}.jpg").write_text("placeholder", encoding="utf-8")
    (root / name / "metadata.json").write_text(
        json.dumps(
            {
                "language_instruction": "put object in bowl",
                "task_id": "put_object",
                "environment_id": "kitchen_01",
                "camera_names": ["main"],
            }
        ),
        encoding="utf-8",
    )
    (root / name / "actions.json").write_text("[]", encoding="utf-8")
    (root / name / "goal.jpg").write_text("placeholder", encoding="utf-8")


def test_existing_manifest_is_normalized(tmp_path):
    subset = tmp_path / "subset"
    subset.mkdir()
    frame_paths = [f"traj_000001/images/frame_{idx:06d}.jpg" for idx in range(24)]
    (subset / "manifest.jsonl").write_text(
        json.dumps({"trajectory_id": "traj_000001", "num_frames": 24, "frame_paths": frame_paths}) + "\n",
        encoding="utf-8",
    )
    summary = build_or_normalize_user_subset_manifest(subset, tmp_path / "run" / "manifest.jsonl")
    assert summary["manifest_exists"] is True
    assert summary["generated_manifest_exists"] is True
    records = load_bridgedata_manifest_jsonl(tmp_path / "run" / "manifest.jsonl")
    assert records[0]["dataset_name"] == "BridgeData V2"
    assert records[0]["trajectory_id"] == "traj_000001"


def test_directory_per_trajectory_autobuilds_manifest_with_metadata(tmp_path):
    subset = tmp_path / "subset"
    _make_traj(subset, "traj_000001", 30)
    summary = build_or_normalize_user_subset_manifest(subset, tmp_path / "run" / "manifest.jsonl")
    assert summary["manifest_source"] == "directory_per_trajectory"
    assert summary["num_manifest_records"] == 1
    records = load_bridgedata_manifest_jsonl(tmp_path / "run" / "manifest.jsonl")
    record = records[0]
    assert record["actions_path"] == "traj_000001/actions.json"
    assert record["language_instruction"] == "put object in bowl"
    assert record["goal_image_path"] == "traj_000001/goal.jpg"
    assert record["metadata"]["source"] == "user_provided_tiny_subset"
