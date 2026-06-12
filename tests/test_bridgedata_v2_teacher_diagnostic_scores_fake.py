import torch

from analysis.bridgedata_v2_teacher_diagnostic_scores import (
    minmax_normalize,
    score_status,
    summarize_label_score_ablation,
    topk_overlap_fraction,
)


def test_step31_score_schema_and_topk_overlap_fake():
    a = minmax_normalize(torch.arange(16 * 392, dtype=torch.float32).reshape(16, 392))
    b = minmax_normalize(torch.flip(a, dims=[0]))
    assert float(a.min()) == 0.0
    assert float(a.max()) == 1.0
    overlap = topk_overlap_fraction(a, b, 64)
    assert 0.0 <= overlap <= 1.0
    summary = summarize_label_score_ablation(
        [
            score_status("proxy_importance", "ok", mean_val_loss=2.0),
            score_status("teacher_occlusion_delta", "ok", mean_val_loss=3.0),
            score_status("gradient_saliency", "skipped", "teacher checkpoint/state_dict not saved by Step30B"),
        ]
    )
    assert summary["best_diagnostic_score"] == "proxy_importance"
    assert summary["teacher_topk_underperforms_proxy_confirmed"] is True
    assert summary["context_utility_claim_allowed"] is False
