from pathlib import Path

import yaml


def test_step37_sources_do_not_import_tensorflow_or_download_or_save_checkpoints():
    paths = [
        "data/bridgedata_v2_factorized_selector_dataset_step37.py",
        "data/bridgedata_v2_factorized_selector_losses_step37.py",
        "data/bridgedata_v2_factorized_selector_metrics_step37.py",
        "models/bridgedata_v2_proxy_factorized_selector.py",
        "training/bridgedata_v2_factorized_selector_trainer_step37.py",
        "scripts/train_bridgedata_v2_factorized_selector_step37.py",
        "scripts/smoke_test_bridgedata_v2_factorized_selector_step37.py",
        "eval/eval_bridgedata_v2_factorized_selector_step37.py",
    ]
    text = "\n".join(Path(path).read_text(encoding="utf-8") for path in paths)
    for forbidden in [
        "import tensorflow",
        "import tensorflow_datasets",
        "tfds.load",
        "requests.get",
        "urlretrieve",
        "snapshot_download",
        "from_pretrained",
        "torch.save(",
        "data/token_shards/",
        "data/importance_shards/",
        "train_current_importance=True",
        "context_utility_claim_allowed=True",
    ]:
        assert forbidden not in text


def test_step37_config_outputs_only_ignored_run_dir_and_docs():
    config = yaml.safe_load(
        Path("configs/bridgedata_v2_tfds_factorized_selector_step37.yaml").read_text(encoding="utf-8")
    )
    output_text = "\n".join(str(value) for value in config["output"].values())
    assert "runs/bridgedata_v2_tfds_factorized_selector_step37_v1" in output_text
    assert "data/token_shards" not in output_text
    assert "data/importance_shards" not in output_text
    assert "checkpoints" not in output_text
    assert config["training"]["save_checkpoint"] is False
    assert config["training"]["save_state_dict"] is False
