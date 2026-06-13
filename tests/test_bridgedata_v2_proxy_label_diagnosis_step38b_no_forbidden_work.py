from pathlib import Path


def test_step38b_sources_do_not_import_tensorflow_or_download_or_train():
    paths = [
        "data/bridgedata_v2_proxy_label_diagnosis_step38b.py",
        "scripts/diagnose_bridgedata_v2_proxy_label_step38b.py",
        "scripts/smoke_test_bridgedata_v2_proxy_label_diagnosis_step38b.py",
        "eval/eval_bridgedata_v2_proxy_label_diagnosis_step38b.py",
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


def test_step38b_sources_do_not_use_action_or_language_as_inputs():
    text = Path("data/bridgedata_v2_proxy_label_diagnosis_step38b.py").read_text(encoding="utf-8")

    assert "action_used_as_input\": False" in text
    assert "language_used_as_input\": False" in text
    assert "future_tokens_loaded_for_diagnosis\": False" in text
