import torch

from models.bridgedata_v2_teacher_diagnostic_variants import (
    DIAGNOSTIC_VARIANTS,
    BridgeDataTeacherDiagnosticPredictor,
    diagnostic_summaries,
    target_summary,
)


def test_step31_diagnostic_variants_forward_fake_tensors():
    context = torch.randn(1, 16, 392, 768)
    current = torch.randn(1, 4, 392, 768)
    future = torch.randn(1, 4, 392, 768)
    importance = torch.rand(1, 16, 392)
    for variant in DIAGNOSTIC_VARIANTS:
        model = BridgeDataTeacherDiagnosticPredictor(variant=variant, hidden_dim=32, topk=8)
        pred = model(context, current, importance)
        assert list(pred.shape) == [1, 768]
        current_summary, context_summary = diagnostic_summaries(context, current, importance, variant, topk=8)
        assert list(current_summary.shape) == [1, 768]
        assert list(context_summary.shape) == [1, 768]
    assert list(target_summary(future, current, "future_delta_last_minus_current").shape) == [1, 768]
