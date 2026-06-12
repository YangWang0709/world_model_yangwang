from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/bridgedata_v2_tfds_token_extraction_step24.yaml")
SCRIPT_PATH = Path("scripts/extract_bridgedata_v2_tfds_videomae_tokens.py")


def test_token_extraction_config_and_script_are_local_only():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["videomae"]["allow_model_download"] is False
    assert config["videomae"]["local_files_only"] is True
    assert config["videomae"]["use_existing_local_model_only"] is True
    assert config["guards"]["no_training"] is True
    assert config["guards"]["no_importance_generation"] is True

    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "allow_download\": False" in source
    assert "local_files_only\": True" in source
    assert "TRANSFORMERS_OFFLINE" in source
    assert "HF_HUB_OFFLINE" in source
    assert "download_videomae_checkpoint" not in source
    assert "from_pretrained(" not in source
