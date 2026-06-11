from pathlib import Path

from scripts.audit_step17_context_bottleneck import REQUIRED_GITIGNORE, forbidden_git_status_entries


def test_gitignore_contains_required_artifact_blacklist():
    text = Path(".gitignore").read_text(encoding="utf-8")
    missing = [item for item in REQUIRED_GITIGNORE if item not in text]
    assert not missing


def test_artifact_blacklist_flags_generated_artifacts_without_false_source_token_hits():
    status = "\n".join(
        [
            "A  scripts/extract_context_tokens.py",
            "A  tests/test_step17_no_current_token_drop_strict.py",
            "A  data/context_token_shards/foo.pt",
            "A  runs/context_bottleneck/foo.json",
            "A  weights/model.bin",
            "A  notes/password.txt",
        ]
    )
    bad = forbidden_git_status_entries(status)
    assert "A  scripts/extract_context_tokens.py" not in bad
    assert "A  tests/test_step17_no_current_token_drop_strict.py" not in bad
    assert "A  data/context_token_shards/foo.pt" in bad
    assert "A  runs/context_bottleneck/foo.json" in bad
    assert "A  weights/model.bin" in bad
    assert "A  notes/password.txt" in bad
