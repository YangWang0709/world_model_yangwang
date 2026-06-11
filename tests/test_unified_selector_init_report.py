from pathlib import Path

import pytest
import torch

from models.attention_selector import AttentionSelector
from models.unified_predictive_importance_selector import (
    UnifiedPredictiveImportanceSelector,
    initialize_context_selector_from_state_checkpoint_with_report,
)
from training.student_selector_trainer import save_student_selector_checkpoint
from training.train_unified_context_selector import save_unified_selector_checkpoint


def _legacy_checkpoint(path: Path, token_dim: int = 8, hidden_dim: int = 4) -> Path:
    config = {"token_dim": token_dim, "hidden_dim": hidden_dim, "task_dim": token_dim, "use_task": False, "dropout": 0.0}
    model = AttentionSelector(**config)
    return save_student_selector_checkpoint(path, model, optimizer=None, step=1, model_config=config)


def test_unified_selector_init_report_compatible_checkpoint(tmp_path: Path):
    ckpt = _legacy_checkpoint(tmp_path / "legacy.pt")
    selector = UnifiedPredictiveImportanceSelector(token_dim=8, hidden_dim=4, dropout=0.0)
    report = initialize_context_selector_from_state_checkpoint_with_report(selector, ckpt)
    assert report["initialized_from_state_selector"] is True
    assert report["loaded_compatible_key_count"] > 0
    assert report["total_legacy_key_count"] >= report["loaded_compatible_key_count"]
    assert report["error"] is None


def test_unified_selector_init_report_incompatible_checkpoint_allow_true(tmp_path: Path):
    ckpt = _legacy_checkpoint(tmp_path / "legacy_bad.pt", token_dim=7, hidden_dim=5)
    selector = UnifiedPredictiveImportanceSelector(token_dim=8, hidden_dim=4, dropout=0.0)
    report = initialize_context_selector_from_state_checkpoint_with_report(
        selector,
        ckpt,
        allow_random_init_if_incompatible=True,
    )
    assert report["initialized_from_state_selector"] is False
    assert report["error"]


def test_unified_selector_init_report_incompatible_checkpoint_allow_false(tmp_path: Path):
    ckpt = _legacy_checkpoint(tmp_path / "legacy_bad.pt", token_dim=7, hidden_dim=5)
    selector = UnifiedPredictiveImportanceSelector(token_dim=8, hidden_dim=4, dropout=0.0)
    with pytest.raises(Exception):
        initialize_context_selector_from_state_checkpoint_with_report(
            selector,
            ckpt,
            allow_random_init_if_incompatible=False,
        )


def test_unified_selector_checkpoint_saves_init_report(tmp_path: Path):
    selector = UnifiedPredictiveImportanceSelector(token_dim=8, hidden_dim=4, dropout=0.0)
    init_report = {
        "initialized_from_state_selector": True,
        "checkpoint_path": "legacy.pt",
        "loaded_compatible_key_count": 3,
        "total_legacy_key_count": 3,
        "missing_keys": [],
        "unexpected_keys": [],
        "incompatible_keys": [],
        "error": None,
    }
    summary = {"init_report": init_report}
    path = save_unified_selector_checkpoint(tmp_path / "unified.pt", selector, None, 1, {"token_dim": 8, "hidden_dim": 4}, summary)
    payload = torch.load(path, map_location="cpu")
    assert payload["init_report"] == init_report

