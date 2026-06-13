from pathlib import Path


def test_step39a_sources_do_not_import_tensorflow_download_train_or_save():
    paths = [
        "data/bridgedata_v2_proxy_label_redesign_step39a.py",
        "scripts/redesign_bridgedata_v2_proxy_label_step39a.py",
        "scripts/smoke_test_bridgedata_v2_proxy_label_redesign_step39a.py",
        "eval/eval_bridgedata_v2_proxy_label_redesign_step39a.py",
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
        ".backward(",
        ".step(",
        "optimizer.step",
        "torch.save(",
        "save_checkpoint=True",
        "save_state_dict=True",
        "data/token_shards/",
        "data/importance_shards/",
        "train_current_importance=True",
        "selector_training_allowed=True",
        "context_utility_claim_allowed=True",
    ]:
        assert forbidden not in text


def test_step39a_sources_do_not_use_action_language_or_future_tokens_as_inputs():
    text = Path("data/bridgedata_v2_proxy_label_redesign_step39a.py").read_text(encoding="utf-8")

    assert '"action_used_as_input": False' in text
    assert '"language_used_as_input": False' in text
    assert '"future_tokens_used_as_input": False' in text
