from pathlib import Path

import yaml

from data.bridgedata_v2_real_tiny_acquisition import acquire_bridgedata_v2_real_tiny_subset


def test_acquisition_safe_stops_without_safe_candidate(tmp_path):
    config = yaml.safe_load(Path("configs/bridgedata_v2_real_tiny_validation_step21.yaml").read_text(encoding="utf-8"))
    config["dataset"]["download_root"] = str(tmp_path / "downloads")
    config["dataset"]["extract_root"] = str(tmp_path / "extracted")
    config["dataset"]["subset_root"] = str(tmp_path / "subset")
    config["output"]["acquisition_summary_json"] = str(tmp_path / "run" / "acquisition_summary.json")
    probe = {
        "safe_candidate_found": False,
        "safe_stop": True,
        "reason": "No official <=1GB tiny sample found or file size unknown.",
    }
    summary = acquire_bridgedata_v2_real_tiny_subset(config, probe)
    assert summary["download_attempted"] is False
    assert summary["download_performed"] is False
    assert summary["safe_stop"] is True
    assert summary["user_provided_subset_required"] is True
    assert not (tmp_path / "downloads").exists()
    assert Path(config["output"]["acquisition_summary_json"]).exists()
