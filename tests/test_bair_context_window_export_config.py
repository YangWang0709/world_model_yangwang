from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_bair_context_window_export_config_bounds_and_env():
    cfg = yaml.safe_load((ROOT / "configs" / "export_bair_context_windows_1000_128.yaml").read_text(encoding="utf-8"))
    assert cfg["dataset"]["train_samples"] == 1000
    assert cfg["dataset"]["test_samples"] == 128
    assert cfg["dataset"]["context_len"] == 8
    assert cfg["dataset"]["current_len"] == 4
    assert cfg["dataset"]["future_len"] == 4
    assert "tgpawb_tfds_py311" in cfg["environment"]["tfds_env"]
    assert cfg["dataset"]["use_action_as_input"] is False
    assert cfg["dataset"]["use_endeffector_as_input"] is False
    assert "download" not in cfg["input"]
