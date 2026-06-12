import json
from pathlib import Path

import yaml

from eval.eval_bridgedata_v2_tfds_field_resolver import evaluate_field_resolver


def _write_config(tmp_path: Path, *, image_field: str, language_field: str = "steps/language_instruction") -> Path:
    config = yaml.safe_load(Path("configs/bridgedata_v2_tfds_field_resolver_step23_5.yaml").read_text(encoding="utf-8"))
    run = tmp_path / "run"
    run.mkdir(parents=True)
    for key, name in {
        "resolved_fields_json": "resolved_fields.json",
        "resolved_manifest_jsonl": "manifest.jsonl",
        "resolved_manifest_summary_json": "manifest_summary.json",
        "resolved_window_manifest_jsonl": "windows.jsonl",
        "resolved_window_summary_json": "window_summary.json",
        "eval_json": "eval.json",
        "eval_md": "eval.md",
    }.items():
        config["output"][key] = str(run / name)
    (run / "resolved_fields.json").write_text(
        json.dumps(
            {
                "image_field": image_field,
                "image_field_valid": image_field.startswith("steps/observation/image_"),
                "image_rejected_fields": [{"field": "episode_metadata/has_image_0", "reason": "metadata flag"}],
                "action_field": "steps/action",
                "language_field": language_field,
                "goal_field": None,
                "download_performed": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "manifest_summary.json").write_text(json.dumps({"num_manifest_records": 10}), encoding="utf-8")
    (run / "window_summary.json").write_text(
        json.dumps(
            {
                "num_windows": 155,
                "training_performed": False,
                "token_extraction_performed": False,
                "importance_generation_performed": False,
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_field_resolver_eval_passes_for_real_image_field(tmp_path):
    docs = tmp_path / "docs"
    summary = evaluate_field_resolver(_write_config(tmp_path, image_field="steps/observation/image_0"), docs_dir=docs)
    assert summary["pass"] is True
    assert summary["recommended_step24"]["name"] == "BridgeData V2 TFDS mini-shard token extraction dry-run"
    report = (docs / "BRIDGEDATA_V2_TFDS_FIELD_RESOLVER_REPORT.md").read_text(encoding="utf-8")
    assert "Field resolver patch passed." in report


def test_field_resolver_eval_fails_for_metadata_image_flag(tmp_path):
    summary = evaluate_field_resolver(_write_config(tmp_path, image_field="episode_metadata/has_image_0"), docs_dir=tmp_path / "docs")
    assert summary["pass"] is False
    assert summary["image_field_is_metadata_flag"] is True


def test_field_resolver_eval_fails_when_embedding_beats_instruction(tmp_path):
    summary = evaluate_field_resolver(
        _write_config(tmp_path, image_field="steps/observation/image_0", language_field="steps/language_embedding"),
        docs_dir=tmp_path / "docs",
    )
    assert summary["pass"] is False
