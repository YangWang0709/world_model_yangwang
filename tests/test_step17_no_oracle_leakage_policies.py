import pytest
import torch

from models.context_selection_policies import select_context_tokens


class FakeSelector:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def eval(self):
        return self

    def __call__(self, *, context_tokens, current_tokens, mode):
        assert mode == "context"
        return self.logits.to(context_tokens.device)


def _batch(importance: torch.Tensor):
    return {"importance_scores_norm": importance}


def test_learned_context_selector_does_not_read_teacher_importance():
    context = torch.randn(2, 20, 4)
    current = torch.randn(2, 5, 4)
    logits = torch.arange(40, dtype=torch.float32).reshape(2, 20)
    selector = FakeSelector(logits)
    a = select_context_tokens(context_tokens=context, current_tokens=current, policy="learned_context_selector_topK", topk=4, selector=selector, batch=_batch(torch.zeros(2, 20)))
    b = select_context_tokens(context_tokens=context, current_tokens=current, policy="learned_context_selector_topK", topk=4, selector=selector, batch=_batch(torch.flip(logits, dims=[1])))
    assert torch.equal(a["selected_indices"], b["selected_indices"])


def test_hybrid_context_selector_does_not_read_teacher_importance():
    context = torch.randn(2, 20, 4)
    current = torch.randn(2, 5, 4)
    logits = torch.arange(40, dtype=torch.float32).reshape(2, 20)
    selector = FakeSelector(logits)
    a = select_context_tokens(context_tokens=context, current_tokens=current, policy="hybrid_context_learned_uniform", topk=6, selector=selector, batch=_batch(torch.zeros(2, 20)), seed=0)
    b = select_context_tokens(context_tokens=context, current_tokens=current, policy="hybrid_context_learned_uniform", topk=6, selector=selector, batch=_batch(torch.ones(2, 20)), seed=0)
    assert torch.equal(a["selected_indices"], b["selected_indices"])


def test_random_and_uniform_context_do_not_read_teacher_importance():
    context = torch.randn(2, 40, 4)
    current = torch.randn(2, 5, 4)
    zeros = _batch(torch.zeros(2, 40))
    ones = _batch(torch.ones(2, 40))
    rand_a = select_context_tokens(context_tokens=context, current_tokens=current, policy="random_context_topK", topk=8, batch=zeros, seed=123)
    rand_b = select_context_tokens(context_tokens=context, current_tokens=current, policy="random_context_topK", topk=8, batch=ones, seed=123)
    rand_c = select_context_tokens(context_tokens=context, current_tokens=current, policy="random_context_topK", topk=8, batch=ones, seed=124)
    assert torch.equal(rand_a["selected_indices"], rand_b["selected_indices"])
    assert not torch.equal(rand_a["selected_indices"], rand_c["selected_indices"])
    uniform_a = select_context_tokens(context_tokens=context, current_tokens=current, policy="uniform_context_topK", topk=8, batch=zeros)
    uniform_b = select_context_tokens(context_tokens=context, current_tokens=current, policy="uniform_context_topK", topk=8, batch=ones)
    assert torch.equal(uniform_a["selected_indices"], uniform_b["selected_indices"])


def test_teacher_context_importance_topk_requires_teacher_labels():
    context = torch.randn(2, 20, 4)
    current = torch.randn(2, 5, 4)
    with pytest.raises(KeyError):
        select_context_tokens(context_tokens=context, current_tokens=current, policy="teacher_context_importance_topK", topk=4, batch={})
    out = select_context_tokens(context_tokens=context, current_tokens=current, policy="teacher_context_importance_topK", topk=4, batch=_batch(torch.arange(40, dtype=torch.float32).reshape(2, 20)))
    assert out["selected_indices"].shape == (2, 4)

