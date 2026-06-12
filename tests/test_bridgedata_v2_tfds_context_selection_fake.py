import torch

from data.bridgedata_v2_tfds_context_selection import select_context_tokens


def _sample():
    importance = torch.arange(16 * 392, dtype=torch.float32).reshape(16, 392)
    return {
        "context_tokens": torch.zeros((16, 392, 768)),
        "current_tokens": torch.zeros((4, 392, 768)),
        "future_tokens": torch.zeros((4, 392, 768)),
        "context_importance": importance,
    }


def test_current_only_selects_no_context_and_keeps_current_full():
    result = select_context_tokens(_sample(), {"name": "current_only", "use_context": False, "topk": 0})
    assert list(result["selected_context_tokens"].shape) == [0, 768]
    assert result["num_selected"] == 0
    assert result["current_tokens_kept_full"] is True


def test_random_and_proxy_topk_select_expected_counts():
    random_result = select_context_tokens(
        _sample(),
        {"name": "random_context_topk", "use_context": True, "selection": "random", "topk": 256, "seed": 42},
    )
    proxy_result = select_context_tokens(
        _sample(),
        {"name": "proxy_importance_topk", "use_context": True, "selection": "importance_topk", "topk": 256},
    )
    assert random_result["num_selected"] == 256
    assert proxy_result["num_selected"] == 256
    assert int(proxy_result["selected_indices"].min()) >= 16 * 392 - 256
    assert proxy_result["selected_importance_mass"] >= random_result["selected_importance_mass"]
    assert "current_importance" not in proxy_result


def test_full_context_reference_selects_every_context_token():
    result = select_context_tokens(
        _sample(),
        {"name": "full_context_reference", "use_context": True, "selection": "full", "topk": None, "deployable": False},
    )
    assert result["num_selected"] == 16 * 392
    assert list(result["selected_context_tokens"].shape) == [16 * 392, 768]
    assert result["deployable"] is False
