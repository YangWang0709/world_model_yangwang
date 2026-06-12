import json
from pathlib import Path

from eval.eval_bridgedata_v2_real_tiny_validation import evaluate_real_tiny_validation


def _write_common(run_dir: Path):
    run_dir.mkdir(parents=True)
    (run_dir / "download_probe_summary.json").write_text(
        json.dumps({"safe_stop": True, "reason": "No official <=1GB tiny sample found or file size unknown.", "download_candidates": []}),
        encoding="utf-8",
    )
    (run_dir / "acquisition_summary.json").write_text(
        json.dumps(
            {
                "safe_stop": True,
                "reason": "No official <=1GB tiny sample found or file size unknown.",
                "download_performed": False,
                "download_bytes": None,
                "full_download_performed": False,
                "large_download_performed": False,
                "training_performed": False,
                "token_extraction_performed": False,
                "importance_generation_performed": False,
            }
        ),
        encoding="utf-8",
    )


def test_eval_recommends_token_extraction_when_real_format_validated(tmp_path):
    run_dir = tmp_path / "run"
    docs_dir = tmp_path / "docs"
    _write_common(run_dir)
    manifest = run_dir / "real_tiny_manifest.jsonl"
    windows = run_dir / "real_tiny_window_manifest.jsonl"
    manifest.write_text("{}", encoding="utf-8")
    windows.write_text("{}", encoding="utf-8")
    (run_dir / "real_tiny_validation_summary.json").write_text(
        json.dumps(
            {
                "real_format_validated": True,
                "safe_stop": False,
                "manifest_path": str(manifest),
                "real_window_manifest_jsonl": str(windows),
                "num_manifest_records": 1,
                "num_valid_trajectories": 1,
                "num_skipped_trajectories": 0,
                "num_windows": 7,
                "frame_count_min": 30,
                "frame_count_mean": 30,
                "frame_count_max": 30,
                "has_images_likely": True,
                "has_actions_likely": False,
                "has_language_likely": False,
                "has_goal_image_likely": False,
                "training_performed": False,
                "token_extraction_performed": False,
                "importance_generation_performed": False,
            }
        ),
        encoding="utf-8",
    )
    summary = evaluate_real_tiny_validation(run_dir, docs_dir)
    assert summary["safety_gate_pass"] is True
    assert summary["real_format_validated"] is True
    assert summary["safe_stop"] is False
    assert summary["recommended_step22"]["name"] == "BridgeData V2 real tiny token extraction dry-run"


def test_eval_safe_stop_does_not_claim_real_validation(tmp_path):
    run_dir = tmp_path / "run"
    docs_dir = tmp_path / "docs"
    _write_common(run_dir)
    (run_dir / "real_tiny_validation_summary.json").write_text(
        json.dumps(
            {
                "real_format_validated": False,
                "safe_stop": True,
                "reason": "No safe official tiny sample could be downloaded.",
                "num_manifest_records": 0,
                "num_valid_trajectories": 0,
                "num_skipped_trajectories": 0,
                "num_windows": 0,
                "training_performed": False,
                "token_extraction_performed": False,
                "importance_generation_performed": False,
            }
        ),
        encoding="utf-8",
    )
    summary = evaluate_real_tiny_validation(run_dir, docs_dir)
    assert summary["safety_gate_pass"] is True
    assert summary["real_format_validated"] is False
    assert summary["safe_stop"] is True
    assert summary["recommended_step22"]["name"] == "User-provided BridgeData tiny subset preparation"
