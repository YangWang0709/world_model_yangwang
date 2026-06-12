from pathlib import Path

from scripts.dryrun_droid_schema import run_droid_dryrun


def test_droid_dryrun_no_download_and_missing_sample_dir_is_ok(tmp_path):
    missing = tmp_path / "missing_sample_dir"
    summary = run_droid_dryrun(output_dir=tmp_path, local_sample_dir=missing)
    assert summary["dataset_name"] == "DROID"
    assert summary["no_download"] is True
    assert summary["full_download_allowed"] is False
    assert summary["full_download_performed"] is False
    assert summary["local_sample_dir_exists"] is False
    assert "proposed_long_context_sample_mapping" in summary
    assert (tmp_path / "droid_dryrun_summary.json").exists()
    assert not list(Path(".").glob("**/droid*.pt"))
