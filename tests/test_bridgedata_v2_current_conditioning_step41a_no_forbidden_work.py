from pathlib import Path


STEP41A_PATHS = [
    "models/bridgedata_v2_current_conditioned_selector_step41a.py",
    "data/bridgedata_v2_current_conditioning_dataset_step41a.py",
    "data/bridgedata_v2_current_conditioning_losses_step41a.py",
    "data/bridgedata_v2_current_conditioning_metrics_step41a.py",
    "training/bridgedata_v2_current_conditioning_trainer_step41a.py",
    "scripts/train_bridgedata_v2_current_conditioning_step41a.py",
    "scripts/smoke_test_bridgedata_v2_current_conditioning_step41a.py",
    "eval/eval_bridgedata_v2_current_conditioning_step41a.py",
]


def test_step41a_sources_do_not_import_tensorflow_download_or_save_artifacts():
    text = "\n".join(Path(path).read_text(encoding="utf-8") for path in STEP41A_PATHS)
    for forbidden in [
        "import tensorflow",
        "import tensorflow_datasets",
        "tfds.load",
        "requests.get",
        "urlretrieve",
        "snapshot_download",
        "from_pretrained",
        "torch.save(",
        "save_checkpoint=True",
        "save_state_dict=True",
        "data/token_shards/",
        "data/importance_shards/",
        "train_current_importance=True",
        "downstream_selector_use_allowed=True",
        "final_selector_training_allowed=True",
        "current_importance_training_allowed=True",
        "context_utility_claim_allowed=True",
    ]:
        assert forbidden not in text


def test_step41a_sources_do_not_use_action_language_or_future_tokens_as_selector_inputs():
    dataset_text = Path("data/bridgedata_v2_current_conditioning_dataset_step41a.py").read_text(encoding="utf-8")
    trainer_text = Path("training/bridgedata_v2_current_conditioning_trainer_step41a.py").read_text(encoding="utf-8")

    assert '"action_used_as_input": False' in dataset_text or '"action_used_as_input": False' in trainer_text
    assert '"language_used_as_input": False' in dataset_text or '"language_used_as_input": False' in trainer_text
    assert '"future_tokens_exposed_to_selector": False' in dataset_text or '"future_tokens_exposed_to_selector": False' in trainer_text
    assert '"current_importance_training_performed": False' in trainer_text
