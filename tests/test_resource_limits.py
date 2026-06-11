"""Tests for Step 9A resource summary helpers."""

from __future__ import annotations

from scripts.check_resource_limits import collect_resource_summary


def test_collect_resource_summary_has_expected_keys(tmp_path) -> None:
    summary = collect_resource_summary(tmp_path)

    for key in (
        "cpu_count",
        "ram_total_gib",
        "ram_available_gib",
        "disk_free_gib",
        "cuda_available",
        "gpu_name",
        "recommended",
        "warnings",
    ):
        assert key in summary
    assert summary["recommended"]["batch_size"] == 2
    assert summary["recommended"]["num_workers"] == 0
    assert summary["recommended"]["max_samples"] <= 100
