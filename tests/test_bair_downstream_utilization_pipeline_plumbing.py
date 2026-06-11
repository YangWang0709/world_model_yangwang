from pathlib import Path

from training.run_bair_downstream_utilization_ablation import (
    is_step16_owned_output_path,
    select_phase_a_top_variants,
    step_names,
    write_partial_summary,
)


def test_step16_runner_step_list():
    assert step_names() == [
        "resource_check",
        "phase_a",
        "select_phase_b",
        "phase_b",
        "write_summary",
    ]


def test_step16_refuses_protected_outputs():
    assert is_step16_owned_output_path("/home/ubuntu22/tgpawb_world_model/runs/downstream_utilization_bair_1000_128_v1")
    assert not is_step16_owned_output_path("/home/ubuntu22/tgpawb_world_model/runs/bair_1000_128_multiseed_validation_v1")
    assert not is_step16_owned_output_path("/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_1000_128_weighted_mse_multiseed_v1")


def test_select_phase_a_top2_ignores_failed_rows():
    rows = [
        {"variant": "bad", "success": False, "student_future_mse": 0.1},
        {"variant": "third", "success": True, "student_future_mse": 1.7},
        {"variant": "first", "success": True, "student_future_mse": 1.5},
        {"variant": "second", "success": True, "student_future_mse": 1.6},
    ]
    assert select_phase_a_top_variants(rows, top_n=2) == ["first", "second"]


def test_partial_summary_records_failure(tmp_path: Path):
    path = write_partial_summary(
        tmp_path,
        current_step="phase_a",
        completed_steps=["resource_check"],
        status="failed",
        error="RuntimeError('boom')",
        selected_phase_b_variants=[],
    )
    text = path.read_text(encoding="utf-8")
    assert "downstream_utilization_bair_1000_128" in text
    assert "boom" in text
