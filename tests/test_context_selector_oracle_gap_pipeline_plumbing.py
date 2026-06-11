from pathlib import Path

from training.context_selector_oracle_gap_trainer import (
    is_step18_owned_output_path,
    step_names,
    variant_run_name,
    write_partial_summary,
)


def test_step18_runner_step_list():
    assert step_names() == [
        "resource_check",
        "validate_step17_inputs",
        "label_diagnostic",
        "phase_a_selectors",
        "phase_a_downstream",
        "phase_b_selectors",
        "phase_b_downstream",
        "write_summary",
    ]


def test_step18_refuses_protected_outputs():
    assert is_step18_owned_output_path("/home/ubuntu22/tgpawb_world_model/runs/context_selector_oracle_gap_bair_1000_128_v1")
    assert not is_step18_owned_output_path("/home/ubuntu22/tgpawb_world_model/runs/context_bottleneck_bair_1000_128_v1")
    assert not is_step18_owned_output_path("/home/ubuntu22/tgpawb_world_model/data/context_token_shards/bair_context_videomae_1000_128")
    assert not is_step18_owned_output_path("/home/ubuntu22/tgpawb_world_model/runs/downstream_utilization_bair_1000_128_v1")


def test_step18_partial_summary_and_variant_names(tmp_path: Path):
    assert variant_run_name("hybrid weighted/mse", 2) == "hybrid_weighted_mse_seed2"
    path = write_partial_summary(
        tmp_path,
        current_step="phase_a_selectors",
        completed_steps=["resource_check", "validate_step17_inputs"],
        status="failed",
        error="RuntimeError('boom')",
    )
    text = path.read_text(encoding="utf-8")
    assert "context_selector_oracle_gap_bair_1000_128" in text
    assert "boom" in text
