"""Plumbing tests for the Step 15 runner without real BAIR/model work."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_bair_1000_128_multiseed_validation import (
    PROTECTED_OUTPUT_MARKERS,
    STEP_DEFINITIONS,
    is_step15_owned_output_path,
    selector_seed_run_name,
    step_names,
    student_seed_run_name,
    validate_step15_output_paths,
    write_partial_summary,
)


def test_step15_runner_step_list_is_ordered() -> None:
    assert step_names() == [
        "resource_check",
        "export_subset",
        "extract_tokens",
        "train_teacher",
        "generate_importance",
        "train_selectors",
        "train_student_world_models",
        "baseline_comparison",
        "write_summary",
    ]
    assert [step.description for step in STEP_DEFINITIONS]


def test_step15_output_path_guard_blocks_earlier_outputs() -> None:
    assert is_step15_owned_output_path("/home/ubuntu22/tgpawb_world_model/runs/teacher_bair_videomae_1000_128_v1")
    for marker in PROTECTED_OUTPUT_MARKERS:
        assert not is_step15_owned_output_path(f"/home/ubuntu22/tgpawb_world_model/runs/{marker}")


def test_validate_step15_output_paths_rejects_step14_run() -> None:
    config = {
        "dataset": {"subset_dir": "/repo/data/bair_robot_pushing_small_subset_1000_128"},
        "tokens": {"output_root": "/repo/data/token_shards/bair_videomae_1000_128"},
        "importance": {"output_root": "/repo/data/importance_shards/bair_videomae_teacher_1000_128"},
        "output": {"run_root": "/repo/runs", "run_name": "bair_1000_128_multiseed_validation_v1"},
        "teacher": {"run_name": "teacher_bair_videomae_500_64_v1"},
        "selector": {"run_root_name": "student_selector_bair_videomae_1000_128_weighted_mse_multiseed_v1"},
        "student_world_model": {"run_root_name": "student_world_model_bair_videomae_1000_128_weighted_mse_multiseed_v1"},
        "baseline": {"run_name": "baseline_comparison_bair_videomae_1000_128_multiseed_v1"},
    }
    with pytest.raises(ValueError):
        validate_step15_output_paths(config)


def test_per_seed_run_names_are_stable() -> None:
    assert selector_seed_run_name(2) == "weighted_mse_alpha2_seed2"
    assert student_seed_run_name(2) == "learned_selector_weighted_mse_alpha2_seed2"


def test_partial_failure_summary_is_written(tmp_path: Path) -> None:
    path = write_partial_summary(
        tmp_path,
        current_step="train_selectors",
        completed_steps=["resource_check", "export_subset", "extract_tokens", "train_teacher", "generate_importance"],
        status="failed",
        error="boom",
        resource={"oom": False},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["stage"] == "bair_1000_128_multiseed_validation"
    assert payload["status"] == "failed"
    assert payload["current_step"] == "train_selectors"
    assert payload["error"] == "boom"
