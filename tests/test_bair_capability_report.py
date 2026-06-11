"""Tests for BAIR capability reporting without downloading data."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.check_bair_dataset_capabilities import build_capability_summary


def test_bair_capability_summary_has_required_keys(tmp_path: Path) -> None:
    config = {
        "project_root": str(tmp_path),
        "dataset": {
            "tfds_name": "bair_robot_pushing_small",
            "tfds_version": "2.0.0",
            "data_dir": str(tmp_path / "tfds"),
            "download": True,
        },
        "resource_limits": {"min_disk_free_gib": 80},
    }
    config_path = tmp_path / "bair.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = build_capability_summary(config_path, write_outputs=False)

    assert "tensorflow_datasets_available" in summary
    assert "disk_free_gib" in summary
    assert "ram_available_gib" in summary
    assert "can_attempt_download" in summary
    assert "blocking_reason" in summary
    assert summary["download_allowed"] is True
