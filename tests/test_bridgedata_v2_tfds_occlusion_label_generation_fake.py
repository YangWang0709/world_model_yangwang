import torch

from data.bridgedata_v2_tfds_occlusion_teacher_labels import compute_occlusion_importance_for_sample
from models.bridgedata_v2_trained_predictor_teacher import CurrentConditionedContextAttentionPredictor


def test_step30b_fake_teacher_generates_context_only_importance():
    torch.manual_seed(0)
    model = CurrentConditionedContextAttentionPredictor(hidden_dim=32, seed=0)
    sample = {
        "sample_id": "fake",
        "context_tokens": torch.randn(16, 392, 768),
        "current_tokens": torch.randn(4, 392, 768),
        "future_tokens": torch.randn(4, 392, 768),
    }
    result = compute_occlusion_importance_for_sample(
        model,
        sample,
        device=torch.device("cpu"),
        occlusion_batch_size=2048,
    )
    assert list(result["context_importance_norm"].shape) == [16, 392]
    assert list(result["temporal_importance"].shape) == [16]
    assert list(result["spatial_importance"].shape) == [392]
    assert float(result["context_importance_norm"].min()) >= 0.0
    assert float(result["context_importance_norm"].max()) <= 1.0
    assert result["stats"]["train_current_importance"] is False
    assert "current_importance" not in result
