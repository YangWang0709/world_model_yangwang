from pathlib import Path

import yaml


def test_step34_source_does_not_import_tensorflow_or_download_assets():
    source_paths = [
        "data/bridgedata_v2_tfds_proxy_temporal_selector_step34.py",
        "eval/eval_bridgedata_v2_tfds_proxy_temporal_selector_step34.py",
        "scripts/prepare_bridgedata_v2_tfds_proxy_temporal_selector_step34.py",
        "scripts/smoke_test_bridgedata_v2_tfds_proxy_temporal_selector_step34.py",
        "models/bridgedata_v2_proxy_temporal_selector.py",
    ]
    text = "\n".join(Path(path).read_text(encoding="utf-8") for path in source_paths)
    for forbidden in [
        "import tensorflow",
        "import tensorflow_datasets",
        "tfds.load",
        "urlretrieve",
        "requests.get",
        "snapshot_download",
        "from_pretrained",
        "optimizer.step(",
        "torch.save(",
        "train_current_importance=True",
        "context_utility_claim_allowed=True",
    ]:
        assert forbidden not in text


def test_step34_outputs_are_run_docs_only_not_data_or_checkpoints():
    config = yaml.safe_load(
        Path("configs/bridgedata_v2_tfds_proxy_temporal_selector_step34.yaml").read_text(encoding="utf-8")
    )
    output_text = "\n".join(str(value) for value in config["output"].values())
    assert "runs/bridgedata_v2_tfds_proxy_temporal_selector_step34_v1" in output_text
    assert "docs/STEP34_PROXY_SUPERVISED_TEMPORAL_SELECTOR_PLAN.md" in output_text
    assert "data/token_shards" not in output_text
    assert "data/importance_shards" not in output_text
    assert "checkpoints" not in output_text
    assert config["selector_scaffold"]["save_checkpoint"] is False
    assert config["selector_scaffold"]["save_state_dict"] is False
