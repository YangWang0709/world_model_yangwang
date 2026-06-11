from pathlib import Path

import yaml


def test_step17_no_current_importance_training_config_and_source():
    config = yaml.safe_load(Path("configs/train_unified_context_selector_bair_1000_128.yaml").read_text(encoding="utf-8"))
    assert config["training"]["train_mode"] == "context"
    assert config["training"]["train_state_mode"] is False
    serialized = Path("configs/train_unified_context_selector_bair_1000_128.yaml").read_text(encoding="utf-8")
    assert "current_importance_shards" not in serialized
    assert "current_importance_scores" not in serialized

    step17 = yaml.safe_load(Path("configs/context_bottleneck_bair_1000_128.yaml").read_text(encoding="utf-8"))
    assert step17["context_bottleneck_world_model"]["current_token_drop"] is False
    assert step17["unified_selector"]["train_state_mode"] is False

    source = Path("training/train_unified_context_selector.py").read_text(encoding="utf-8")
    assert 'target = batch["importance_scores_norm"]' in source
    assert '"trained_current_importance": False' in source
    assert "current_importance_scores" not in source

