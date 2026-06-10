from __future__ import annotations

import json
from pathlib import Path

from eval.eval_baseline_comparison import evaluate_baseline_comparison, load_baseline_rows


def _write_summary(run_dir: Path, policy: str, seed: int, mse: float, coverage: float) -> None:
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
        "teacher_future_mse": 0.1,
        "student_teacher_gap": mse - 0.1,
        "student_teacher_ratio": mse / 0.1,
        "token_retention_ratio": 0.25,
        "selected_top1_hit_rate": coverage,
        "selected_topk_hit_rate": coverage,
        "selected_key_coverage": coverage,
        "checkpoint_path": str(subrun / "checkpoints" / "student_world_model_step_000003.pt"),
        "device": "cpu",
    }
    (subrun / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_eval_baseline_comparison_regenerates_summaries(tmp_path: Path) -> None:
    _write_summary(tmp_path, "random_k", 0, 0.3, 0.25)
    _write_summary(tmp_path, "oracle_key", 0, 0.1, 1.0)
    _write_summary(tmp_path, "learned_selector", 0, 0.2, 1.0)

    rows = load_baseline_rows(tmp_path)
    result = evaluate_baseline_comparison(tmp_path)

    assert len(rows) == 3
    assert (tmp_path / "baseline_summary.json").exists()
    assert (tmp_path / "baseline_summary.csv").exists()
    assert (tmp_path / "baseline_summary.md").exists()
    assert result["aggregate"]["sanity_gate"]["pass"] is True
