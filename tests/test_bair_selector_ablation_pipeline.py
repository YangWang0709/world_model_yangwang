from __future__ import annotations

import json
from pathlib import Path

from eval.eval_bair_selector_ablation import (
    load_selector_ablation_rows,
    write_selector_ablation_summaries,
)
from eval.eval_bair_selector_downstream_ablation import (
    load_downstream_ablation_rows,
    write_downstream_ablation_summaries,
)


def _selector_summary(run_dir: Path, variant: str, loss_type: str, mse: float, importance: float) -> None:
    subrun = run_dir / "selectors" / f"{variant}_seed0"
    subrun.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant_name": variant,
        "variant_run_name": f"{variant}_seed0",
        "loss_type": loss_type,
        "seed": 0,
        "success": True,
        "final_loss": mse,
        "test_importance_mse": mse,
        "test_importance_mae": mse,
        "test_pearson_corr_mean": 0.5,
        "test_target_top1_overlap": 0.1,
        "test_target_topk_overlap": 0.2,
        "test_selected_teacher_importance_mean": importance,
        "test_random_teacher_importance_mean": 0.4,
        "test_selected_vs_random_importance_gap": importance - 0.4,
    }
    (subrun / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _downstream_summary(run_dir: Path, variant: str, mse: float, importance: float) -> None:
    subrun = run_dir / "downstream_student_world_models" / f"{variant}_seed0"
    subrun.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant_name": variant,
        "variant_run_name": f"{variant}_seed0",
        "loss_type": variant,
        "seed": 0,
        "success": True,
        "final_loss": mse,
        "student_future_mse": mse,
        "teacher_mse": 0.1,
        "student_teacher_ratio": mse / 0.1,
        "token_retention_ratio": 16 / 392,
        "selector_target_top1_overlap": 0.1,
        "selector_target_topk_overlap": 0.2,
        "selected_teacher_importance_mean": importance,
        "random_teacher_importance_mean": 0.4,
        "selected_vs_random_importance_gap": importance - 0.4,
    }
    (subrun / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_selector_ablation_eval_pipeline_from_fake_summaries(tmp_path: Path) -> None:
    _selector_summary(tmp_path, "mse_only", "mse_only", 0.3, 0.5)
    _selector_summary(tmp_path, "hybrid", "hybrid_weighted_mse_rank_bce", 0.2, 0.7)
    _downstream_summary(tmp_path, "mse_only", 0.62, 0.5)
    _downstream_summary(tmp_path, "hybrid", 0.50, 0.7)
    step12 = tmp_path / "step12.json"
    step12.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "policy": "learned_selector",
                        "student_future_mse": 0.58,
                        "selected_teacher_importance_mean": 0.6,
                        "selector_target_topk_overlap": 0.16,
                    },
                    {
                        "policy": "uniform_k",
                        "student_future_mse": 0.55,
                        "selected_teacher_importance_mean": 0.48,
                        "selector_target_topk_overlap": 0.06,
                    },
                    {
                        "policy": "teacher_importance_topk",
                        "student_future_mse": 0.30,
                        "selected_teacher_importance_mean": 0.90,
                        "selector_target_topk_overlap": 1.0,
                    },
                    {
                        "policy": "random_k",
                        "student_future_mse": 0.70,
                        "selected_teacher_importance_mean": 0.42,
                        "selector_target_topk_overlap": 0.04,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    selector_rows = load_selector_ablation_rows(tmp_path)
    selector_summary = write_selector_ablation_summaries(selector_rows, tmp_path)
    downstream_rows = load_downstream_ablation_rows(tmp_path)
    downstream_summary = write_downstream_ablation_summaries(
        downstream_rows,
        tmp_path,
        step12_summary_json=step12,
        selector_summary={"rows": selector_rows, "aggregate": selector_summary["selector_aggregate"]},
    )

    assert (tmp_path / "selector_ablation_summary.json").exists()
    assert (tmp_path / "downstream_ablation_summary.json").exists()
    assert (tmp_path / "selector_ablation_combined_report.json").exists()
    assert downstream_summary["combined_report"]["sanity_gate"]["pass"] is True
