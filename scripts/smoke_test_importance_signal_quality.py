"""End-to-end Step 5.5 smoke test for structured toy importance signal quality."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.importance_shards import load_importance_shard, summarize_importance_shard
from eval.eval_importance_quality import evaluate_importance_quality
from scripts.create_structured_token_shards import config_from_mapping, load_yaml as load_toy_yaml
from scripts.generate_predictive_importance import generate_predictive_importance, load_yaml as load_importance_yaml
from training.teacher_trainer import run_teacher_training
from training.train_teacher import load_yaml as load_teacher_yaml


REPORT_PATH = PROJECT_ROOT / "docs" / "IMPORTANCE_SIGNAL_QUALITY_REPORT.md"
DIAGNOSTIC_PATH = PROJECT_ROOT / "docs" / "STEP5_5_IMPORTANCE_SIGNAL_DIAGNOSTIC.md"


def _project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def _rewrite_structured_toy_config(config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    updated["data"] = dict(updated.get("data", {}))
    updated["output"] = dict(updated.get("output", {}))
    token_dir = _project_path("data", "token_shards", "structured_toy")
    updated["data"]["token_shard_dir"] = str(token_dir)
    updated["output"]["token_shard_dir"] = str(token_dir)
    updated["output"]["summary_path"] = str(token_dir / "structured_token_summary.json")
    updated["output"]["overwrite"] = True
    return updated


def _rewrite_teacher_config(config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    updated["data"] = dict(updated["data"])
    updated["output"] = dict(updated["output"])
    updated["data"]["token_shard_dir"] = str(_project_path("data", "token_shards", "structured_toy"))
    updated["output"]["run_root"] = str(_project_path("runs"))
    updated["output"]["run_name"] = "teacher_structured_toy_tiny_v1"
    return updated


def _rewrite_importance_config(config: dict[str, Any], checkpoint_path: str) -> dict[str, Any]:
    updated = dict(config)
    updated["data"] = dict(updated["data"])
    updated["teacher"] = dict(updated["teacher"])
    updated["output"] = dict(updated["output"])
    output_dir = _project_path("data", "importance_shards", "structured_toy_teacher")
    updated["data"]["token_shard_dir"] = str(_project_path("data", "token_shards", "structured_toy"))
    updated["teacher"]["checkpoint"] = checkpoint_path
    updated["output"]["output_dir"] = str(output_dir)
    updated["output"]["summary_path"] = str(output_dir / "importance_summary.json")
    updated["output"]["overwrite"] = True
    return updated


def _clean_outputs() -> None:
    for path in (
        _project_path("data", "token_shards", "structured_toy"),
        _project_path("data", "importance_shards", "structured_toy_teacher"),
        _project_path("runs", "teacher_structured_toy_tiny_v1"),
    ):
        if path.exists():
            shutil.rmtree(path)


def _write_report(
    token_summary: dict[str, Any],
    training_summary: dict[str, Any],
    importance_summary: dict[str, Any],
    first_shard_summary: dict[str, Any],
    quality_metrics: dict[str, Any],
    pass_flag: bool,
) -> None:
    command = (
        "/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab "
        "python scripts/smoke_test_importance_signal_quality.py"
    )
    report = [
        "# Importance Signal Quality Report",
        "",
        f"Command: `{command}`",
        "",
        f"- structured token shard path: `{token_summary['output_dir']}`",
        f"- structured teacher checkpoint: `{training_summary['checkpoint_path']}`",
        f"- structured importance output path: `{importance_summary['output_dir']}`",
        f"- generated importance shard count: `{importance_summary['num_importance_shards']}`",
        f"- generated importance shard files: `{importance_summary['importance_shard_files']}`",
        "",
        "## Training Summary",
        "",
        "```json",
        json.dumps(training_summary, indent=2),
        "```",
        "",
        "## First Importance Shard Summary",
        "",
        "```json",
        json.dumps(first_shard_summary, indent=2),
        "```",
        "",
        "## Quality Metrics",
        "",
        "```json",
        json.dumps(quality_metrics, indent=2),
        "```",
        "",
        f"IMPORTANCE_SIGNAL_QUALITY_PASS = {str(pass_flag).lower()}",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_diagnostic(
    token_summary: dict[str, Any],
    training_summary: dict[str, Any],
    importance_summary: dict[str, Any],
    first_shard_summary: dict[str, Any],
    quality_metrics: dict[str, Any],
    pass_flag: bool,
) -> None:
    diagnostic = [
        "# STEP5.5 Importance Signal Diagnostic Report",
        "",
        "## 1. Why Step 5.5 Was Needed",
        "",
        "Step 5 passed the predictive-importance pipeline smoke test, but the dummy toy setup produced near-all-negative raw importance and all-zero normalized importance. That means the pipeline works mechanically, while the toy signal is not suitable yet for training a Student selector.",
        "",
        "## 2. Root Cause Analysis",
        "",
        "- Dummy encoder tokens are weakly related or unrelated to visual content.",
        "- Mean pooling dilutes the effect of one token across 196 tokens.",
        "- Zero masking can reduce noise instead of increasing prediction loss.",
        "- The dummy future target lacks a clear causal dependency on local past tokens.",
        "- The tiny teacher objective can be too easy, so single-token occlusion has a very small effect.",
        "",
        "## 3. Structured Token Toy Setup",
        "",
        f"- num_samples: `{token_summary['num_samples']}`",
        f"- num_tokens: `{token_summary['num_tokens']}`",
        f"- token_dim: `{token_summary['token_dim']}`",
        f"- num_key_tokens: `{token_summary['num_key_tokens']}`",
        f"- signal_scale: `{token_summary['signal_scale']}`",
        f"- noise_scale: `{token_summary['noise_scale']}`",
        "- key labels: `metadata.key_token_indices` and optional `aux_labels.key_token_mask`",
        "",
        "## 4. Teacher Training",
        "",
        f"- run dir: `{training_summary['run_dir']}`",
        f"- checkpoint: `{training_summary['checkpoint_path']}`",
        f"- num steps: `{training_summary['num_steps']}`",
        f"- initial loss: `{training_summary['initial_loss']}`",
        f"- final loss: `{training_summary['final_loss']}`",
        f"- best loss: `{training_summary['best_loss']}`",
        f"- loss_decreased: `{training_summary['loss_decreased']}`",
        "",
        "## 5. Importance Generation",
        "",
        f"- output dir: `{importance_summary['output_dir']}`",
        f"- generated shards: `{importance_summary['num_importance_shards']}`",
        f"- importance mean/std/min/max: `{first_shard_summary['importance_mean']} / {first_shard_summary['importance_std']} / {first_shard_summary['importance_min']} / {first_shard_summary['importance_max']}`",
        f"- normalized importance mean/std/min/max: `{first_shard_summary['importance_norm_mean']} / {first_shard_summary['importance_norm_std']} / {first_shard_summary['importance_norm_min']} / {first_shard_summary['importance_norm_max']}`",
        "",
        "## 6. Importance Quality Evaluation",
        "",
        f"- positive_importance_ratio: `{quality_metrics['positive_importance_ratio']}`",
        f"- key_token_importance_mean: `{quality_metrics['key_token_importance_mean']}`",
        f"- non_key_token_importance_mean: `{quality_metrics['non_key_token_importance_mean']}`",
        f"- key_vs_non_key_gap: `{quality_metrics['key_vs_non_key_gap']}`",
        f"- top1_hit_rate: `{quality_metrics['top1_hit_rate']}`",
        f"- topk_hit_rate: `{quality_metrics['topk_hit_rate']}`",
        f"- pass: `{pass_flag}`",
        "",
        "## 7. Smoke Test Result",
        "",
        f"`IMPORTANCE_SIGNAL_QUALITY_PASS = {str(pass_flag).lower()}`. See `docs/IMPORTANCE_SIGNAL_QUALITY_REPORT.md`.",
        "",
        "## 8. Pytest Result",
        "",
        "Run `python -m pytest tests -q` after this smoke test. The final local summary records the observed result.",
        "",
        "## 9. Git Commit",
        "",
        "Branch: `feature/tgpawb-step3-token-extraction`. Commit hash and push status are recorded after git commit/push. No PR is created in this stage.",
        "",
        "## 10. What Was Not Done",
        "",
        "- no real dataset download",
        "- no real V-JEPA / VideoMAE encoder",
        "- no large model download",
        "- no long training",
        "- no student training",
        "- no VLM grounding",
        "- no generated `.pt` committed",
        "- no checkpoint committed",
        "- no password or token saved",
        "- no PR created",
        "",
        "## 11. Next Step Recommendation",
        "",
        "Step 6 should train a Student attention selector against the structured toy importance labels first, verifying that it can recover key tokens before moving to less controlled data.",
    ]
    DIAGNOSTIC_PATH.write_text("\n".join(diagnostic) + "\n", encoding="utf-8")


def main() -> None:
    _clean_outputs()

    toy_config = _rewrite_structured_toy_config(load_toy_yaml(PROJECT_ROOT / "configs" / "structured_token_toy.yaml"))
    toy_dataclass, token_dir, overwrite = config_from_mapping(toy_config)
    from data.structured_token_toy import write_structured_token_shards

    token_summary = write_structured_token_shards(token_dir, toy_dataclass, overwrite=overwrite)

    teacher_config = _rewrite_teacher_config(load_teacher_yaml(PROJECT_ROOT / "configs" / "train_teacher_structured_toy.yaml"))
    training_summary = run_teacher_training(teacher_config)

    importance_config = _rewrite_importance_config(
        load_importance_yaml(PROJECT_ROOT / "configs" / "generate_importance_structured_toy.yaml"),
        training_summary["checkpoint_path"],
    )
    importance_summary = generate_predictive_importance(importance_config)
    importance_dir = Path(importance_summary["output_dir"])
    first_shard = load_importance_shard(importance_dir / importance_summary["importance_shard_files"][0])
    first_shard_summary = summarize_importance_shard(first_shard)

    quality_metrics = evaluate_importance_quality(token_dir, importance_dir)
    eval_path = importance_dir / "eval_importance_summary.json"
    eval_path.write_text(json.dumps(quality_metrics, indent=2), encoding="utf-8")

    checks = {
        "positive_importance_ratio": quality_metrics["positive_importance_ratio"] > 0,
        "importance_std": quality_metrics["importance_std"] > 0,
        "normalized_importance_std": quality_metrics["normalized_importance_std"] > 0,
        "key_gt_non_key": quality_metrics["key_token_importance_mean"] > quality_metrics["non_key_token_importance_mean"],
        "topk_hit_rate": quality_metrics["topk_hit_rate"] >= 0.5,
    }
    pass_flag = all(checks.values())
    quality_metrics["quality_gates"] = checks
    quality_metrics["pass"] = pass_flag
    eval_path.write_text(json.dumps(quality_metrics, indent=2), encoding="utf-8")

    _write_report(token_summary, training_summary, importance_summary, first_shard_summary, quality_metrics, pass_flag)
    _write_diagnostic(token_summary, training_summary, importance_summary, first_shard_summary, quality_metrics, pass_flag)
    print(REPORT_PATH.read_text(encoding="utf-8"))
    if not pass_flag:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
