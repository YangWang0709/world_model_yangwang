import json
from pathlib import Path

from data.bridgedata_v2_manifest_autobuilder import build_manifest_from_directory
from data.bridgedata_v2_manifest_schema import load_bridgedata_manifest_jsonl


def _make_traj(root: Path, name: str, num_frames: int):
    traj = root / name
    images = traj / "images"
    images.mkdir(parents=True)
    for idx in range(num_frames):
        (images / f"frame_{idx:06d}.jpg").write_text("placeholder", encoding="utf-8")
    (traj / "metadata.json").write_text(
        json.dumps(
            {
                "language_instruction": "place object",
                "task_id": "fake_task",
                "environment_id": "fake_env",
                "camera_names": ["main"],
            }
        ),
        encoding="utf-8",
    )
    (traj / "actions.json").write_text("[]", encoding="utf-8")
    (traj / "goal.jpg").write_text("placeholder", encoding="utf-8")


def test_autobuilder_keeps_long_trajectories_and_skips_short(tmp_path):
    subset = tmp_path / "subset"
    _make_traj(subset, "traj_long", 30)
    _make_traj(subset, "traj_short", 10)
    manifest = tmp_path / "run" / "manifest.jsonl"
    summary = build_manifest_from_directory(subset, manifest, min_frames=24)
    assert summary["manifest_built"] is True
    assert summary["num_manifest_records"] == 1
    assert summary["num_valid_trajectories"] == 1
    assert summary["num_skipped_trajectories"] == 1
    records = load_bridgedata_manifest_jsonl(manifest)
    assert records[0]["trajectory_id"] == "traj_long"
    assert records[0]["num_frames"] == 30
    assert len(records[0]["frame_paths"]) == 30
