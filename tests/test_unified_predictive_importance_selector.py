import torch

from models.context_selection_policies import topk_from_retention
from models.unified_predictive_importance_selector import UnifiedPredictiveImportanceSelector
from training.train_unified_context_selector import weighted_mse_alpha


def test_unified_selector_context_mode_conditions_on_current_and_backward():
    selector = UnifiedPredictiveImportanceSelector(token_dim=768, hidden_dim=64, condition_on_current=True)
    context = torch.randn(2, 784, 768)
    current = torch.randn(2, 392, 768)
    target = torch.rand(2, 784)
    logits = selector(context_tokens=context, current_tokens=current, mode="context")
    assert logits.shape == (2, 784)
    loss = weighted_mse_alpha(torch.sigmoid(logits), target, alpha=2.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert topk_from_retention(784, 0.04081632653061224) == 32


def test_unified_selector_state_legacy_interface_exists_and_current_not_dropped():
    selector = UnifiedPredictiveImportanceSelector(token_dim=16, hidden_dim=8)
    tokens = torch.randn(2, 10, 16)
    current = torch.randn(2, 4, 16)
    legacy_logits = selector(tokens=tokens, mode="state_legacy")
    context_logits = selector(context_tokens=tokens, current_tokens=current, mode="context")
    assert legacy_logits.shape == (2, 10)
    assert context_logits.shape == (2, 10)
    assert current.shape == (2, 4, 16)
