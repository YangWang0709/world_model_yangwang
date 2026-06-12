from pathlib import Path

import yaml

from scripts.build_bridgedata_v2_context_windows import build_from_config


def test_builder_summary_records_no_download_no_training_no_token_or_importance(tmp_path):
    config = yaml.safe_load(Path("configs/bridgedata_v2_tiny_window_builder_step20.yaml").read_text(encoding="utf-8"))
    missing = tmp_path / "missing_subset"
    run_dir = tmp_path / "run"
    config["dataset"]["local_subset_candidates"] = [str(missing)]
    config["dataset"]["manifest_candidates"] = [str(missing / "manifest.jsonl")]
    config["output_records"]["output_root"] = str(run_dir)
    config["output_records"]["window_manifest_jsonl"] = str(run_dir / "bridgedata_v2_window_manifest.jsonl")
    config["output_records"]["trajectory_summary_json"] = str(run_dir / "trajectory_summary.json")
    config["output_records"]["builder_summary_json"] = str(run_dir / "bridgedata_v2_window_builder_summary.json")
    config["output_records"]["builder_summary_md"] = str(run_dir / "bridgedata_v2_window_builder_summary.md")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = build_from_config(path)
    assert config["dataset"]["full_download_allowed"] is False
    assert summary["full_download_performed"] is False
    assert summary["large_download_performed"] is False
    assert summary["training_performed"] is False
    assert summary["token_extraction_performed"] is False
    assert summary["importance_generation_performed"] is False
    bad_suffixes = {".tfrecord", ".hdf5", ".h5", ".pt", ".pth", ".npy", ".npz"}
    assert not [path for path in run_dir.rglob("*") if path.suffix.lower() in bad_suffixes]
