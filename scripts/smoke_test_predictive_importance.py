"""End-to-end smoke test for Step 5 predictive token importance."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.importance_shards import load_importance_shard, summarize_importance_shard, validate_importance_shard
from eval.eval_importance_summary import evaluate_importance_dir
from scripts.generate_predictive_importance import generate_predictive_importance, load_yaml
from scripts.inspect_importance_shard import inspect_importance_shard
from scripts.smoke_test_teacher_training import main as run_teacher_training_smoke
from scripts.smoke_test_token_extraction import main as run_token_extraction_smoke


REPORT_PATH = PROJECT_ROOT / "docs" / "PREDICTIVE_IMPORTANCE_SMOKE_REPORT.md"


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "generate_importance_dummy.yaml"
    config = load_yaml(config_path)
    token_dir = Path(config["data"]["token_shard_dir"])
    token_glob = config["data"].get("shard_glob", "tokens_shard_*.pt")
    if not list(token_dir.glob(token_glob)):
        run_token_extraction_smoke()

    teacher_checkpoint = Path(config["teacher"]["checkpoint"])
    if not teacher_checkpoint.exists():
        run_teacher_training_smoke()

    output_dir = Path(config["output"]["output_dir"])
    if output_dir.exists():
        shutil.rmtree(output_dir)

    generation_summary = generate_predictive_importance(config)
    shard_paths = sorted(output_dir.glob("importance_shard_*.pt"))
    first_shard = load_importance_shard(shard_paths[0], map_location="cpu")
    validate_importance_shard(first_shard, strict=True)
    first_summary = summarize_importance_shard(first_shard)

    eval_summary = evaluate_importance_dir(output_dir)
    inspect_summary = inspect_importance_shard(shard_paths[0])
    checks = {
        "shard_count_ok": len(shard_paths) >= 2,
        "importance_scores_shape_ok": first_summary["importance_scores_shape"][1] == 196,
        "importance_scores_norm_shape_ok": (
            first_summary["importance_scores_norm_shape"] == first_summary["importance_scores_shape"]
        ),
        "base_losses_shape_ok": first_summary["base_losses_shape"] == [first_summary["num_samples"]],
        "masked_losses_shape_ok": (
            first_summary["masked_losses_shape"] == first_summary["importance_scores_shape"]
        ),
        "sample_ids_len_ok": len(first_shard["sample_ids"]) == first_summary["num_samples"],
        "inspect_matches_first": inspect_summary["importance_scores_shape"] == first_summary["importance_scores_shape"],
        "eval_samples_ok": int(eval_summary["num_samples"]) >= 16,
    }
    pass_flag = all(checks.values())

    command = (
        "/home/ubuntu22/miniconda3/bin/conda run -n env_isaaclab "
        "python scripts/smoke_test_predictive_importance.py"
    )
    report = [
        "# Predictive Importance Smoke Report",
        "",
        f"Command: `{command}`",
        "",
        f"- teacher checkpoint: `{teacher_checkpoint}`",
        f"- token shard input dir: `{token_dir}`",
        f"- importance output dir: `{output_dir}`",
        f"- generated importance shard count: `{len(shard_paths)}`",
        f"- generated importance shard files: `{[path.name for path in shard_paths]}`",
        f"- validation result: `{pass_flag}`",
        "",
        "## First Importance Shard Summary",
        "",
        "```json",
        json.dumps(first_summary, indent=2),
        "```",
        "",
        "## Eval Importance Summary",
        "",
        "```json",
        json.dumps(eval_summary, indent=2),
        "```",
        "",
        "## Generation Summary",
        "",
        "```json",
        json.dumps(
            {
                "num_source_shards": generation_summary["num_source_shards"],
                "num_importance_shards": generation_summary["num_importance_shards"],
                "device": generation_summary["device"],
                "importance_shard_files": generation_summary["importance_shard_files"],
            },
            indent=2,
        ),
        "```",
        "",
        f"PREDICTIVE_IMPORTANCE_SMOKE_PASS = {str(pass_flag).lower()}",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    if not pass_flag:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
