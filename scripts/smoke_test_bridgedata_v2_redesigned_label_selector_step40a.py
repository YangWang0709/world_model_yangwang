"""Fake-tensor smoke gate for Step40A redesigned-label selector training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_redesigned_label_selector_dataset_step40a import (
    OPTIMIZER_SCOPE,
    attach_targets,
    batch_step40a_samples,
    build_global_spatial_prior_removed_residual_target,
    build_train_split_global_spatial_prior,
    make_step40a_sample_from_tensors,
)
from data.bridgedata_v2_redesigned_label_selector_losses_step40a import redesigned_label_selector_mse_loss
from data.bridgedata_v2_redesigned_label_selector_metrics_step40a import build_step40a_gate_decision
from models.bridgedata_v2_proxy_patch_token_selector import ProxyPatchTokenSelectorHead
from training.bridgedata_v2_redesigned_label_selector_trainer_step40a import _optimizer_scope


def main() -> None:
    torch.manual_seed(42)
    train_samples = [_sample(f"train{i}", scale=1.0 + i * 0.01) for i in range(3)]
    val_samples = [_sample(f"val{i}", scale=3.0 + i) for i in range(2)]
    prior_payload = build_train_split_global_spatial_prior(train_samples)
    train_prior = prior_payload["prior"]
    val_only_prior = build_train_split_global_spatial_prior(val_samples)["prior"]
    if torch.allclose(train_prior, val_only_prior):
        raise AssertionError("train prior unexpectedly matches val-only prior")
    target = build_global_spatial_prior_removed_residual_target(train_samples[0]["original_importance"], train_prior)
    if list(target.shape) != [4, 5] or float(target.min()) < 0.0 or float(target.max()) > 1.0:
        raise AssertionError("bad redesigned target")
    samples = attach_targets(train_samples[:2], train_prior)
    batch = batch_step40a_samples(samples, torch.device("cpu"))
    model = ProxyPatchTokenSelectorHead(
        token_dim=6,
        hidden_dim=8,
        context_frames=4,
        spatial_tokens=5,
        condition_on_current_summary=True,
        use_temporal_embedding=True,
        use_spatial_embedding=True,
        detach_token_inputs=True,
    )
    logits = model(batch["context_tokens"], batch["current_tokens"])
    if list(logits.shape) != [2, 4, 5]:
        raise AssertionError("bad selector output shape")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scope = _optimizer_scope(model, optimizer)
    if scope["optimizer_scope"] != OPTIMIZER_SCOPE:
        raise AssertionError("optimizer scope escaped selector head")
    loss = redesigned_label_selector_mse_loss(logits, batch["target_label"], {"use_sigmoid_scores": True})["loss"]
    if not bool(torch.isfinite(loss)):
        raise AssertionError("loss is not finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    gate = build_step40a_gate_decision(
        within=_passing_metric(),
        cross=_passing_metric(),
        mixed=_passing_metric(),
        leakage={"no_language_or_trajectory_leakage": True},
        requirements={"top256_overlap_mean_min": 0.12, "overfit_gap_max": 0.05},
    )
    if not gate["redesigned_label_selector_smoke_pass"]:
        raise AssertionError("fake passing metrics should pass smoke gate")
    for flag in (
        "downstream_selector_use_allowed",
        "final_selector_training_allowed",
        "current_importance_training_allowed",
        "context_utility_claim_allowed",
        "checkpoint_saved",
        "state_dict_saved",
    ):
        if bool(gate.get(flag, False)):
            raise AssertionError(f"{flag} must remain false")
    print(
        json.dumps(
            {
                "BRIDGEDATA_V2_TFDS_REDESIGNED_LABEL_SELECTOR_STEP40A_PASS": True,
                "optimizer_scope": scope["optimizer_scope"],
                "target_shape": list(target.shape),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _sample(sample_id: str, *, scale: float) -> dict:
    return make_step40a_sample_from_tensors(
        sample_id=sample_id,
        trajectory_id=f"traj_{sample_id}",
        shard_id="shard1",
        data_package_id="fake",
        context_tokens=torch.randn(4, 5, 6),
        current_tokens=torch.randn(2, 5, 6),
        original_importance=torch.rand(4, 5) * float(scale),
    )


def _passing_metric() -> dict:
    return {
        "num_rows": 1,
        "selector_beats_random_token_baseline": True,
        "selector_beats_uniform_or_mean_baseline": True,
        "selector_beats_temporal_broadcast_baseline": True,
        "selector_beats_train_global_spatial_prior_baseline": True,
        "top256_overlap": 0.2,
        "overfit_gap": 0.01,
    }


if __name__ == "__main__":
    main()
