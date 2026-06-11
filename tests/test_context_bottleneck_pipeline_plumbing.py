from pathlib import Path

from training.run_bair_context_bottleneck_validation import (
    is_step17_owned_output_path,
    step_names,
    write_partial_summary,
)


def test_step17_runner_step_list():
    assert step_names() == [
        "resource_check",
        "export_context_windows",
        "extract_context_tokens",
        "train_context_teacher",
        "generate_context_importance",
        "train_unified_context_selector",
        "train_context_bottleneck_world_model",
        "baseline_comparison",
        "write_summary",
    ]


def test_step17_refuses_protected_outputs():
    assert is_step17_owned_output_path("/home/ubuntu22/tgpawb_world_model/runs/context_bottleneck_bair_1000_128_v1")
    assert is_step17_owned_output_path("/home/ubuntu22/tgpawb_world_model/data/context_token_shards/bair_context_videomae_1000_128")
    assert not is_step17_owned_output_path("/home/ubuntu22/tgpawb_world_model/runs/downstream_utilization_bair_1000_128_v1")
    assert not is_step17_owned_output_path("/home/ubuntu22/tgpawb_world_model/runs/student_selector_bair_videomae_1000_128_weighted_mse_multiseed_v1")


def test_step17_partial_summary_records_failure(tmp_path: Path):
    path = write_partial_summary(
        tmp_path,
        current_step="extract_context_tokens",
        completed_steps=["resource_check", "export_context_windows"],
        status="failed",
        error="RuntimeError('boom')",
    )
    text = path.read_text(encoding="utf-8")
    assert "context_bottleneck_bair_1000_128" in text
    assert "boom" in text
