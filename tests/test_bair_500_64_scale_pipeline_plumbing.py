"""Plumbing tests for the Step 14 runner without real BAIR/model work."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_bair_500_64_scale_validation import (
    PROTECTED_OUTPUT_MARKERS,
    STEP_DEFINITIONS,
    is_step14_owned_output_path,
    step_names,
    validate_step14_output_paths,
    write_partial_summary,
)


def test_step14_runner_step_list_is_ordered() -> None:
    assert step_names() == [
        "resource_check",
        "export_subset",
        "extract_tokens",
        "train_teacher",
        "generate_importance",
        "train_selector",
        "train_student_world_model",
        "baseline_comparison",
        "write_summary",
    ]
    assert [step.description for step in STEP_DEFINITIONS]


def test_step14_output_path_guard_blocks_step11_to_step13_outputs() -> None:
    assert is_step14_owned_output_path("/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_500_64_v1")
    for marker in PROTECTED_OUTPUT_MARKERS:
        assert not is_step14_owned_output_path(f"/home/ubuntu22/tgpawb_world_model/runs/{marker}")


def test_validate_step14_output_paths_rejects_smoke_run() -> None:
    config = {
        "dataset": {"subset_dir": "/repo/data/bair_robot_pushing_small_subset_500_64"},
        "tokens": {"output_root": "/repo/data/token_shards/bair_videomae_500_64"},
        "importance": {"output_root": "/repo/data/importance_shards/bair_videomae_teacher_500_64"},
        "output": {"run_root": "/repo/runs", "run_name": "bair_500_64_scale_validation_v1"},
        "teacher": {"run_name": "teacher_bair_videomae_smoke_v1"},
        "selector": {"run_name": "student_selector_bair_videomae_500_64_weighted_mse_v1"},
        "student_world_model": {"run_name": "student_world_model_bair_videomae_500_64_weighted_mse_v1"},
        "baseline": {"run_name": "baseline_comparison_bair_videomae_500_64_v1"},
    }
    with pytest.raises(ValueError):
        validate_step14_output_paths(config)


def test_partial_failure_summary_is_written(tmp_path: Path) -> None:
    path = write_partial_summary(
        tmp_path,
        current_step="extract_tokens",
        completed_steps=["resource_check", "export_subset"],
        status="failed",
        error="boom",
        resource={"oom": False},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["current_step"] == "extract_tokens"
    assert payload["completed_steps"] == ["resource_check", "export_subset"]
    assert payload["error"] == "boom"
