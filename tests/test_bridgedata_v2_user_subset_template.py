from pathlib import Path

from scripts.create_bridgedata_v2_user_subset_template import create_user_subset_template


def test_template_creator_writes_text_only_files(tmp_path):
    summary = create_user_subset_template(tmp_path / "template")
    written = [Path(path) for path in summary["files_written"]]
    assert {path.name for path in written} == {
        "manifest_template.jsonl",
        "metadata_template.json",
        "README_USER_SUBSET_FORMAT.md",
        "example_tree.txt",
    }
    assert all(path.exists() for path in written)
    assert summary["text_only"] is True
    assert not list((tmp_path / "template").glob("**/*.jpg"))
    assert not list((tmp_path / "template").glob("**/*.pt"))
    assert not list((tmp_path / "template").glob("**/*.npy"))
    readme = (tmp_path / "template" / "README_USER_SUBSET_FORMAT.md").read_text(encoding="utf-8")
    assert "data/bridgedata_v2_tiny_user_subset/" in readme
    assert "actions.npy or actions.json optional" in readme
