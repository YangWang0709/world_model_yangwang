import json
from pathlib import Path

from data.bridgedata_v2_tfds_longer_horizon_proxy_importance import summarize_longer_horizon_proxy_importance


def test_proxy_importance_summary_groups_by_horizon_without_current_importance(tmp_path: Path):
    windows = [
        {"sample_id": "gap0_a", "trajectory_id": "traj0", "horizon_gap": 0},
        {"sample_id": "gap4_b", "trajectory_id": "traj1", "horizon_gap": 4},
    ]
    importance = [
        _importance("gap0_a", "traj0"),
        _importance("gap4_b", "traj1"),
    ]
    window_manifest = tmp_path / "windows.jsonl"
    importance_manifest = tmp_path / "importance.jsonl"
    window_manifest.write_text("\n".join(json.dumps(item) for item in windows) + "\n", encoding="utf-8")
    importance_manifest.write_text("\n".join(json.dumps(item) for item in importance) + "\n", encoding="utf-8")
    summary = summarize_longer_horizon_proxy_importance(importance_manifest, window_manifest)
    assert summary["limited_proxy_importance_generation_performed"] is True
    assert summary["horizons"] == [0, 4]
    assert summary["context_importance_shape_example"] == [16, 392]
    assert summary["train_current_importance"] is False
    assert summary["current_importance_generated"] is False
    assert summary["data_importance_shards_written"] is False


def _importance(sample_id: str, trajectory_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "trajectory_id": trajectory_id,
        "importance_artifact_path": f"/tmp/{sample_id}.pt",
        "method": "proxy_token_mse_dryrun",
        "context_importance_shape": [16, 392],
        "temporal_importance_shape": [16],
        "spatial_importance_shape": [392],
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
    }

