import torch

from data.bridgedata_v2_tfds_context_selection import select_context_tokens
from models.bridgedata_v2_context_bottleneck_smoke_model import (
    BridgeDataContextBottleneckSmokePredictor,
    forward_loss_for_policy,
)


def test_smoke_model_forward_returns_finite_loss_for_all_policies(tmp_path):
    sample = {
        "sample_id": "sample",
        "trajectory_id": "traj",
        "context_tokens": torch.zeros((16, 392, 768)),
        "current_tokens": torch.ones((4, 392, 768)),
        "future_tokens": torch.ones((4, 392, 768)),
        "context_importance": torch.linspace(0.0, 1.0, 16 * 392).reshape(16, 392),
    }
    policies = [
        {"name": "current_only", "use_context": False, "topk": 0},
        {"name": "random_context_topk", "use_context": True, "selection": "random", "topk": 256, "seed": 42},
        {"name": "proxy_importance_topk", "use_context": True, "selection": "importance_topk", "topk": 256},
        {"name": "full_context_reference", "use_context": True, "selection": "full", "topk": None},
    ]
    model = BridgeDataContextBottleneckSmokePredictor(hidden_dim=32, seed=7)
    for policy in policies:
        selection = select_context_tokens(sample, policy)
        result = forward_loss_for_policy(sample, selection, model)
        assert result["loss_finite"] is True
        assert result["pred_shape"] == [768]
        assert result["optimizer_step_performed"] is False
        assert result["training_performed"] is False
        assert result["random_init_result_not_scientific"] is True
    assert not list(tmp_path.glob("*.pt"))
