"""Structured stronger-teacher planning for Step28.

This module is deliberately pure: it reads no artifacts, downloads nothing, and
does not import training code. Step28 uses it to turn the Step24-27 diagnosis
into a concrete Step29/Step30 route.
"""

from __future__ import annotations

from typing import Any


PRIMARY_STEP29_NAME = "BridgeData V2 TFDS 32/64-window train-val context utility sanity"
PRIMARY_STEP29_RECOMMENDATION = (
    "Expand token extraction to 32/64 windows from existing TFDS shard and run train/val context utility sanity"
)
ALTERNATIVE_STEP29_NAME = "trained-predictor occlusion teacher dry-run"


def build_teacher_plan(sample_count: int = 4) -> dict[str, Any]:
    """Return candidate stronger-teacher plans and the recommended order."""

    candidate_plans = [
        {
            "id": "A",
            "name": "stronger trained predictor occlusion teacher",
            "summary": (
                "Train a small predictor on more BridgeData token samples, then compute context-token "
                "occlusion delta loss using the trained predictor."
            ),
            "advantages": [
                "Closer to true predictive utility than the mean-token proxy.",
                "Labels come from a trained predictive model rather than a simple proxy.",
            ],
            "limitations": [
                "Requires more samples.",
                "Requires a train/val split.",
                "Costs more than the proxy dry-run.",
            ],
            "allowed_now": False,
            "recommended_stage": "Step30",
        },
        {
            "id": "B",
            "name": "train/val context utility experiment first",
            "summary": (
                "Expand from 4 windows to 32 or 64 windows, train a tiny predictor on a train split, "
                "and evaluate held-out context utility."
            ),
            "advantages": [
                "Tests whether context helps generalization before improving labels.",
                "Avoids spending selector effort on labels that may not help prediction.",
            ],
            "limitations": [
                "Requires Step29 token extraction expansion to more windows.",
                "Still remains a small sanity experiment, not final-scale training.",
            ],
            "allowed_now": False,
            "recommended_stage": "Step29",
        },
        {
            "id": "C",
            "name": "true temporal VideoMAE token extraction",
            "summary": "Replace frame-repeat tokenization with true temporal clip tokens in a later ablation.",
            "advantages": [
                "Encodes temporal structure more faithfully.",
                "May make context/future relations easier to measure.",
            ],
            "limitations": [
                "Changes token shape or extraction behavior.",
                "Needs a separate ablation before conclusions are mixed with Step24 results.",
            ],
            "allowed_now": False,
            "recommended_stage": "after Step29 or Step30",
        },
        {
            "id": "D",
            "name": "current importance diagnostic only",
            "summary": (
                "Compute current-token occlusion sensitivity as a diagnostic, but do not train a current "
                "selector and do not compress current tokens."
            ),
            "advantages": [
                "Can answer whether current tokens are redundant.",
                "Does not disrupt the context bottleneck main line.",
            ],
            "limitations": [
                "Should not become the main objective yet.",
                "Can confuse the context bottleneck conclusion if treated as selector training.",
            ],
            "allowed_now": False,
            "recommended_stage": "diagnostic after context utility is validated",
        },
        {
            "id": "E",
            "name": "scale TFDS windows before stronger teacher",
            "summary": "Use the existing downloaded shard and Step23 windows to increase from 4 to 32/64 windows.",
            "advantages": [
                "Does not require a new shard download.",
                "Creates a better train/val sanity test before selector training.",
            ],
            "limitations": [
                "Still comes from the same shard, so diversity is limited.",
                "Requires a bounded Step29 token extraction expansion.",
            ],
            "allowed_now": False,
            "recommended_stage": "Step29",
        },
    ]
    return {
        "recommended_primary_next_step": (
            "Step29 expand token extraction to 32 windows and run train/val context utility sanity"
        ),
        "recommended_primary_next_step_name": PRIMARY_STEP29_NAME,
        "recommended_primary_next_step_detail": PRIMARY_STEP29_RECOMMENDATION,
        "recommended_secondary_next_step": "Step30 trained-predictor occlusion teacher",
        "alternative_step29": ALTERNATIVE_STEP29_NAME,
        "sample_count": int(sample_count),
        "do_not_train_current_importance_yet": True,
        "train_current_importance_now": False,
        "candidate_plans": candidate_plans,
    }


def build_next_step_decision(sample_count: int = 4) -> dict[str, Any]:
    """Return the explicit Step29 decision emitted by Step28."""

    return {
        "decision": "scale_existing_shard_train_val_context_utility_first",
        "recommended_step29": {
            "name": PRIMARY_STEP29_NAME,
            "condition": "Step28 shows 4-sample overfit is memorization-prone",
            "scope": "use existing shard first; allow limited token extraction expansion, not full training",
            "detail": PRIMARY_STEP29_RECOMMENDATION,
        },
        "alternative_step29": {
            "name": ALTERNATIVE_STEP29_NAME,
            "condition": "if user chooses teacher-first path",
            "scope": "no selector training yet",
        },
        "recommended_step30": {
            "name": "trained-predictor occlusion teacher",
            "condition": "after train/val context utility is measurable on more than 4 windows",
            "scope": "small trained predictor teacher labels; still no selector training",
        },
        "do_not_train_current_importance_yet": True,
        "do_not_train_selector_yet": True,
        "do_not_claim_context_utility_from_4_sample_overfit": True,
        "sample_count": int(sample_count),
        "rationale": (
            "All policies, including current_only, overfit 4 samples. The proxy label ranks concentrated "
            "context tokens, but predictive utility needs held-out validation before selector or current "
            "importance training."
        ),
    }

