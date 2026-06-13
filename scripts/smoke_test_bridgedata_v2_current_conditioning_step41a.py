"""Fake-tensor smoke gate for Step41A current-conditioning diagnosis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_current_conditioning_dataset_step41a import OPTIMIZER_SCOPE
from data.bridgedata_v2_current_conditioning_losses_step41a import current_conditioning_mse_loss
from data.bridgedata_v2_current_conditioning_metrics_step41a import build_step41a_gate_decision
from models.bridgedata_v2_current_conditioned_selector_step41a import VARIANT_NAMES, build_current_conditioned_selector
from training.bridgedata_v2_current_conditioning_trainer_step41a import _optimizer_scope


def main() -> None:
    torch.manual_seed(42)
    config = _fake_config()
    context = torch.randn(2, 4, 5, 6)
    current = torch.randn(2, 2, 5, 6)
    target = torch.rand(2, 4, 5)
    variant_shapes = {}
    optimizer_scopes = {}
    for variant in VARIANT_NAMES:
        model = build_current_conditioned_selector(config, variant)
        output = model(context, current)
        if list(output["scores"].shape) != [2, 4, 5]:
            raise AssertionError(f"bad output shape for {variant}: {tuple(output['scores'].shape)}")
        if variant == "no_current_context_only":
            no_current_output = model(context, None)
            if not torch.allclose(output["scores"], no_current_output["scores"]):
                raise AssertionError("no_current_context_only must not depend on current tokens")
            if output["diagnostics"].get("uses_current_tokens"):
                raise AssertionError("no_current_context_only diagnostics must say current is unused")
        if variant == "current_coarse_spatial_query_attention":
            if int(output["diagnostics"].get("max_attention_elements_seen", 0)) <= 0:
                raise AssertionError("coarse query attention did not report chunked attention elements")
        loss = current_conditioning_mse_loss(output, target, {"use_sigmoid_scores": True})["loss"]
        if not bool(torch.isfinite(loss)):
            raise AssertionError(f"loss is not finite for {variant}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scope = _optimizer_scope(model, optimizer)
        if scope["optimizer_scope"] != OPTIMIZER_SCOPE:
            raise AssertionError(f"optimizer scope escaped Step41A head for {variant}: {scope}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        variant_shapes[variant] = list(output["scores"].shape)
        optimizer_scopes[variant] = scope["optimizer_scope"]

    gate = build_step41a_gate_decision(
        variant_comparison=_passing_variant_comparison(),
        leakage={"no_language_or_trajectory_leakage": True},
        requirements={"top256_overlap_mean_min": 0.12, "overfit_gap_max": 0.05},
    )
    if not gate["current_conditioning_candidate_ready"]:
        raise AssertionError("fake passing metrics should make current-conditioning candidate ready")
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
                "BRIDGEDATA_V2_TFDS_CURRENT_CONDITIONING_STEP41A_PASS": True,
                "variant_shapes": variant_shapes,
                "optimizer_scopes": optimizer_scopes,
                "current_conditioning_candidate_ready": gate["current_conditioning_candidate_ready"],
                "downstream_selector_use_allowed": gate["downstream_selector_use_allowed"],
                "final_selector_training_allowed": gate["final_selector_training_allowed"],
                "current_importance_training_allowed": gate["current_importance_training_allowed"],
                "context_utility_claim_allowed": gate["context_utility_claim_allowed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _fake_config() -> dict:
    return {
        "model": {
            "token_dim": 6,
            "hidden_dim": 8,
            "attention_dim": 4,
            "context_frames": 4,
            "current_frames": 2,
            "spatial_tokens": 5,
            "dropout": 0.0,
        },
        "current_conditioning_variants": {
            "attention_dim": 4,
            "coarse_current_bins": 3,
            "chunk_context_tokens": 4,
            "attention_temperature": 1.0,
            "detach_token_inputs": True,
            "use_temporal_embedding": True,
            "use_spatial_embedding": True,
        },
        "data": {"token_dim": 6, "context_frames": 4, "current_frames": 2, "spatial_tokens": 5},
    }


def _passing_variant_comparison() -> dict:
    metric = {
        "num_rows": 1,
        "selector_val_mse": 0.05,
        "selector_beats_random_token_baseline": True,
        "selector_beats_uniform_or_mean_baseline": True,
        "selector_beats_temporal_broadcast_baseline": True,
        "selector_beats_train_global_spatial_prior_baseline": True,
        "beats_no_current_context_only": True,
        "beats_current_mean_summary": True,
        "top256_overlap": 0.2,
        "overfit_gap": 0.01,
    }
    return {
        "best_variant": "current_coarse_spatial_query_attention",
        "per_variant": {
            "current_coarse_spatial_query_attention": {
                "overall": metric,
                "within_shard": metric,
                "cross_shard": metric,
                "mixed_shard": metric,
            }
        },
    }


if __name__ == "__main__":
    main()
