import torch
from torch import nn

from training.train_context_bottleneck_world_model import _aggregate, evaluate_full_context_teacher_reference


class OffsetTeacher(nn.Module):
    def forward(self, context_tokens: torch.Tensor, current_tokens: torch.Tensor) -> torch.Tensor:
        del context_tokens
        return current_tokens.mean(dim=1) + 1.0


def test_full_context_teacher_reference_helper_and_aggregate():
    current = torch.zeros(2, 3, 4)
    future = torch.zeros(2, 5, 4)
    loader = [
        {
            "context_tokens": torch.zeros(2, 6, 4),
            "current_tokens": current,
            "future_tokens": future,
        }
    ]
    row = evaluate_full_context_teacher_reference(OffsetTeacher(), loader, torch.device("cpu"), seed=0)
    assert row["policy"] == "full_context_teacher_reference"
    assert row["is_teacher_reference"] is True
    assert row["trainable_student"] is False
    assert row["deployable_policy"] is False
    assert row["context_topK"] is None
    assert row["context_retention_ratio"] is None
    assert row["selected_context_importance_mean"] is None
    assert row["selector_target_topk_overlap"] is None
    assert row["future_mse"] == 1.0

    aggregate = _aggregate(
        [
            row,
            {
                "success": True,
                "policy": "current_only",
                "seed": 0,
                "student_future_mse": 1.5,
                "selected_context_importance_mean": None,
                "selector_target_topk_overlap": None,
            },
        ]
    )
    teacher_row = next(item for item in aggregate["aggregate"] if item["policy"] == "full_context_teacher_reference")
    assert teacher_row["selected_context_importance_mean"] is None
    assert teacher_row["selector_target_topk_overlap_mean"] is None
    assert teacher_row["is_teacher_reference"] is True
    assert teacher_row["deployable_policy"] is False

