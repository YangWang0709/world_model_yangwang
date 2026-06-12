import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_user_subset_ingestion import evaluate_user_subset_ingestion


def _write_config_and_common(tmp_path: Path, *, pending: bool, validated: bool) -> Path:
    config = yaml.safe_load(Path("configs/bridgedata_v2_user_subset_ingestion_step22.yaml").read_text(encoding="utf-8"))
    run = tmp_path / "run"
    docs = tmp_path / "docs"
    config["output"]["subset_inspection_json"] = str(run / "inspection.json")
    config["output"]["manifest_summary_json"] = str(run / "manifest_summary.json")
    config["output"]["validation_summary_json"] = str(run / "validation.json")
    config["output"]["window_builder_summary_json"] = str(run / "window_summary.json")
    config["output"]["eval_json"] = str(run / "eval.json")
    config["output"]["eval_md"] = str(run / "eval.md")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run.mkdir(parents=True)
    docs.mkdir(parents=True)
    (run / "inspection.json").write_text(json.dumps({"pending_user_data": pending}), encoding="utf-8")
    (run / "manifest_summary.json").write_text(json.dumps({"num_manifest_records": 0 if pending else 1}), encoding="utf-8")
    (run / "validation.json").write_text(
        json.dumps({"real_format_validated": validated, "pending_user_data": pending}),
        encoding="utf-8",
    )
    (run / "window_summary.json").write_text(
        json.dumps(
            {
                "safety_gate_pass": True,
                "user_subset_exists": not pending,
                "pending_user_data": pending,
                "real_format_validated": validated,
                "manifest_exists": not pending,
                "generated_manifest_exists": not pending,
                "user_window_manifest_exists": validated,
                "num_manifest_records": 0 if pending else 1,
                "num_valid_trajectories": 0 if pending else 1,
                "num_skipped_trajectories": 0,
                "num_windows": 0 if pending else 7,
                "download_performed": False,
                "training_performed": False,
                "token_extraction_performed": False,
                "importance_generation_performed": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_pending_report_does_not_claim_validation(tmp_path):
    config_path = _write_config_and_common(tmp_path, pending=True, validated=False)
    docs_dir = tmp_path / "docs"
    summary = evaluate_user_subset_ingestion(config_path, docs_dir=docs_dir)
    report = (docs_dir / "BRIDGEDATA_V2_USER_SUBSET_INGESTION_REPORT.md").read_text(encoding="utf-8")
    assert summary["pending_user_data"] is True
    assert summary["real_format_validated"] is False
    assert "User subset is not available yet." in report
    assert "Real format validated." not in report


def test_validated_report_recommends_token_dry_run(tmp_path):
    config_path = _write_config_and_common(tmp_path, pending=False, validated=True)
    docs_dir = tmp_path / "docs"
    summary = evaluate_user_subset_ingestion(config_path, docs_dir=docs_dir)
    report = (docs_dir / "BRIDGEDATA_V2_USER_SUBSET_INGESTION_REPORT.md").read_text(encoding="utf-8")
    assert summary["pending_user_data"] is False
    assert summary["real_format_validated"] is True
    assert "Real format validated." in report
    assert "BridgeData V2 user-subset token extraction dry-run" in report
