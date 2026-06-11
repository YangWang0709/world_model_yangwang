import torch

from eval.eval_context_importance_diagnostic import build_context_importance_diagnostic


def test_context_importance_diagnostic_fake_diffuse_labels():
    raw = torch.linspace(-0.01, 0.01, steps=10 * 784).reshape(10, 784)
    norm = torch.full((10, 784), 0.5)
    summary = build_context_importance_diagnostic(
        raw,
        norm,
        topk_values=[8, 16, 32],
        context_topk=32,
        temporal_blocks=8,
        random_trials=2,
        step17_reference={
            "teacher_context_importance_topk_mse": 1.2,
            "random_context_topk_mse": 1.4,
        },
    )
    assert summary["num_samples"] == 10
    assert summary["num_context_tokens"] == 784
    assert "top32_mass_ratio" in summary["topk_concentration"]
    assert len(summary["temporal_block_stats"]["block_stats"]) == 8
    assert summary["oracle_random_gap"]["oracle_vs_random_downstream_mse_gap"] == 0.19999999999999996
    assert summary["label_quality_warnings"]

