import torch

from data.bridgedata_v2_tfds_true_temporal_proxy_importance import (
    METHOD,
    compute_true_temporal_context_importance,
    summarize_true_temporal_importance,
    validate_true_temporal_importance_record,
)


def test_true_temporal_proxy_importance_is_shape_flexible_and_context_only():
    result = compute_true_temporal_context_importance(
        torch.randn(7, 768),
        torch.randn(768),
        torch.randn(768),
    )
    assert result["method"] == METHOD
    assert list(result["context_importance_norm"].shape) == [7]
    assert result["temporal_spatial_summary_available"] is False
    assert result["stats"]["train_current_importance"] is False


def test_true_temporal_proxy_importance_records_temporal_bins_when_available():
    result = compute_true_temporal_context_importance(
        torch.randn(392, 768),
        torch.randn(768),
        torch.randn(768),
    )
    assert result["temporal_spatial_summary_available"] is True
    assert list(result["temporal_importance"].shape) == [2]
    assert list(result["spatial_importance"].shape) == [196]
    record = {
        "sample_id": "s0",
        "trajectory_id": "traj0",
        "importance_artifact_path": "/tmp/ignored.pt",
        "method": METHOD,
        "context_importance_shape": [392],
        "temporal_importance_shape": [2],
        "spatial_importance_shape": [196],
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "current_importance_generated": False,
        "selector_training_performed": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
    }
    assert validate_true_temporal_importance_record(record) is True
    assert summarize_true_temporal_importance([record])["context_importance_shape_example"] == [392]
