from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_context_token_extraction_uses_local_videomae_and_context_output():
    cfg = yaml.safe_load((ROOT / "configs" / "token_extraction_bair_context_videomae_1000_128.yaml").read_text(encoding="utf-8"))
    assert cfg["encoder"]["allow_download"] is False
    assert cfg["encoder"]["local_files_only"] is True
    assert cfg["fallback"]["allow_dummy_fallback"] is False
    assert cfg["encoder"]["model_name_or_path"].startswith("/home/ubuntu22/tgpawb_world_model/model_cache/")
    assert "/data/context_token_shards/" in cfg["extraction"]["output_root"]
    assert cfg["encoder"]["context_num_frames"] == 8
    assert cfg["encoder"]["short_num_frames"] == 4
