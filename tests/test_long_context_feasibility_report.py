import json

from eval.eval_long_context_dataset_feasibility import write_feasibility_summary


def test_feasibility_report_from_fake_dryrun_summaries(tmp_path):
    matrix = {
        "stage": "long_context_dataset_feasibility_step19",
        "no_download": True,
        "resource_snapshot": {"disk_free_gb": 159.0, "gpu_summary": "fake"},
        "datasets": [
            {
                "dataset": "BridgeData V2",
                "priority": "high",
                "full_download_allowed": False,
                "local_cache_exists": False,
                "supports_long_context_likely": True,
                "supports_goal_image_likely": True,
                "supports_language_likely": True,
                "supports_action_likely": True,
                "supports_multicam_likely": None,
                "estimated_local_difficulty": "medium",
                "recommended_role": "first local long-context migration target",
                "recommended_next_action": "metadata-only dry-run",
            }
        ],
    }
    bridge = {
        "no_download": True,
        "recommended_initial_subset": {"trajectories": "100 to 500"},
        "recommended_next_step": "BridgeData V2 tiny-subset context window builder",
    }
    droid = {
        "no_download": True,
        "recommended_role": {
            "first_local_migration_target": False,
            "use_as_future_large_scale_validation": True,
            "require_storage_planning": True,
            "likely_cloud_or_external_disk_for_full_use": True,
        },
        "recommended_next_step": "future large-scale validation",
    }
    (tmp_path / "dataset_candidate_matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
    (tmp_path / "bridgedata_v2_dryrun_summary.json").write_text(json.dumps(bridge), encoding="utf-8")
    (tmp_path / "droid_dryrun_summary.json").write_text(json.dumps(droid), encoding="utf-8")
    docs_dir = tmp_path / "docs"
    summary = write_feasibility_summary(run_dir=tmp_path, docs_dir=docs_dir)
    assert summary["dataset_recommendation"]["first_migration_target"]
    assert summary["cloud_required_now"] is False
    assert summary["full_download_performed"] is False
    assert summary["training_performed"] is False
    assert summary["sanity_gate_pass"] is True
    assert (docs_dir / "LONG_CONTEXT_DATASET_FEASIBILITY_REPORT.md").exists()
