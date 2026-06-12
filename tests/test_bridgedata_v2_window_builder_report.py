from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_window_builder import evaluate_window_builder
from scripts.build_bridgedata_v2_context_windows import build_from_config


def test_eval_report_from_fake_builder_outputs(tmp_path):
    config = yaml.safe_load(Path("configs/bridgedata_v2_tiny_window_builder_step20.yaml").read_text(encoding="utf-8"))
    missing = tmp_path / "missing_subset"
    run_dir = tmp_path / "run"
    docs_dir = tmp_path / "docs"
    config["dataset"]["local_subset_candidates"] = [str(missing)]
    config["dataset"]["manifest_candidates"] = [str(missing / "manifest.jsonl")]
    config["output_records"]["output_root"] = str(run_dir)
    config["output_records"]["window_manifest_jsonl"] = str(run_dir / "bridgedata_v2_window_manifest.jsonl")
    config["output_records"]["trajectory_summary_json"] = str(run_dir / "trajectory_summary.json")
    config["output_records"]["builder_summary_json"] = str(run_dir / "bridgedata_v2_window_builder_summary.json")
    config["output_records"]["builder_summary_md"] = str(run_dir / "bridgedata_v2_window_builder_summary.md")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    build_from_config(path)
    report = evaluate_window_builder(run_dir=run_dir, docs_dir=docs_dir)
    assert report["pass"] is True
    assert report["recommended_step21"]["name"]
    assert report["recommended_step21"]["no_training"] is True
    assert report["training_performed"] is False
    assert report["full_download_performed"] is False
    assert (docs_dir / "BRIDGEDATA_V2_TINY_WINDOW_BUILDER_REPORT.md").exists()
