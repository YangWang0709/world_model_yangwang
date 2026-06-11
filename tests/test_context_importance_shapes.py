import torch

from models.context_bottleneck_world_model import ContextTeacherWorldModel
from scripts.generate_context_predictive_importance import context_importance_for_batch


class RecordingTeacher(ContextTeacherWorldModel):
    def __init__(self):
        super().__init__(token_dim=8, hidden_dim=16, output_dim=8)
        self.current_snapshots = []

    def forward(self, context_tokens, current_tokens):
        self.current_snapshots.append(current_tokens.detach().clone())
        return super().forward(context_tokens, current_tokens)


def test_context_importance_occludes_context_only_and_keeps_current():
    teacher = RecordingTeacher()
    context = torch.randn(2, 12, 8)
    current = torch.randn(2, 6, 8)
    future = torch.randn(2, 6, 8)
    out = context_importance_for_batch(
        teacher,
        context,
        current,
        future,
        token_chunk_size=5,
        mask_value=0.0,
    )
    assert out["importance_scores"].shape == (2, 12)
    assert out["importance_scores_norm"].shape == (2, 12)
    assert out["masked_losses"].shape == (2, 12)
    assert torch.isfinite(out["importance_scores_norm"]).all()
    for snapshot in teacher.current_snapshots:
        assert torch.equal(snapshot, current)
