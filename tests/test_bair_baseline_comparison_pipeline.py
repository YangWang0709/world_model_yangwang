from __future__ import annotations

import json
from pathlib import Path

from eval.eval_bair_baseline_comparison import evaluate_baseline_comparison
from eval.eval_baseline_comparison import load_baseline_rows


def _write_summary(run_dir: Path, policy: str, seed: int, mse: float, importance: float) -> None:
    subrun = run_dir / f"{policy}_seed{seed}"
    subrun.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": policy,
        "seed": seed,
        "num_steps": 3,
        "initial_loss": mse + 0.1,
        "final_loss": mse,
        "best_loss": mse,
        "loss_decreased": True,
        "student_future_mse": mse,
        "teacher_mse": 0.1,
        "teacher_future_mse": 0.1,
        "student_teacher_gap": mse - 0.1,
        "student_teacher_ratio": mse / 0.1,
        "token_retention_ratio": 0.040816,
        "selector_target_top1_overlap": 0.0,
        "selector_target_topk_overlap": 0.25,
        "selected_teacher_importance_mean": importance,
        "random_teacher_importance_mean": 0.4,
        "selected_vs_random_importance_gap": importance - 0.4,
        "checkpoint_path": str(subrun / "checkpoints" / "student_world_model_step_000003.pt"),
        "device": "cpu",
    }
    (subrun / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_bair_eval_baseline_comparison_regenerates_summaries(tmp_path: Path) -> None:
    _write_summary(tmp_path, "random_k", 0, 0.5, 0.40)
    _write_summary(tmp_path, "random_k", 1, 0.7, 0.44)
    _write_summary(tmp_path, "teacher_importance_topk", 0, 0.3, 0.90)
    _write_summary(tmp_path, "learned_selector", 0, 0.6, 0.60)

    rows = load_baseline_rows(tmp_path)
    result = evaluate_baseline_comparison(tmp_path)

    assert len(rows) == 4
    assert (tmp_path / "baseline_summary.json").exists()
    assert (tmp_path / "baseline_summary.csv").exists()
    assert (tmp_path / "baseline_summary.md").exists()
    assert result["aggregate"]["sanity_gate"]["pass"] is True
