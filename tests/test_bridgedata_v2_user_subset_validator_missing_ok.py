from data.bridgedata_v2_user_subset_ingestion import inspect_user_subset_directory
from data.bridgedata_v2_user_subset_validator import validate_user_subset_manifest


def test_missing_user_subset_is_pending_not_failure(tmp_path):
    missing = tmp_path / "missing_subset"
    inspection = inspect_user_subset_directory(missing)
    validation = validate_user_subset_manifest([], missing, user_subset_exists=False)
    assert inspection["user_subset_exists"] is False
    assert inspection["pending_user_data"] is True
    assert validation["pending_user_data"] is True
    assert validation["real_format_validated"] is False
    assert validation["blocking_errors"] == []
    assert validation["safety_gate_pass"] is True
