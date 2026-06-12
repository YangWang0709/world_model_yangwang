"""Step31 failure-analysis decision helpers."""

from __future__ import annotations

from typing import Any


def build_teacher_failure_analysis(
    architecture: dict[str, Any],
    label_scores: dict[str, Any],
    temporal: dict[str, Any],
    current_dominance: dict[str, Any],
    step30b_eval: dict[str, Any],
) -> dict[str, Any]:
    best_arch = str(architecture.get("best_architecture_variant") or "")
    best_score = str(label_scores.get("best_diagnostic_score") or "")
    current_level = str(current_dominance.get("current_dominance_level") or "unknown")
    full_noise = bool(temporal.get("full_context_noise_confirmed") or current_dominance.get("full_context_noise_penalty_present"))
    teacher_under_proxy = bool(label_scores.get("teacher_topk_underperforms_proxy_confirmed"))
    proxy_prior_helped = bool(architecture.get("proxy_prior_helped_attention"))
    delta_gain = bool(temporal.get("delta_target_increases_context_gain"))
    likely_causes = []
    if teacher_under_proxy:
        likely_causes.append("fixed-attention occlusion label is less useful than Step30A proxy topK on held-out sanity")
    if proxy_prior_helped:
        likely_causes.append("teacher architecture benefits from proxy-prior context scoring")
    if current_level in ("high", "medium"):
        likely_causes.append("short-horizon future summary is current-dominant")
    if full_noise:
        likely_causes.append("full context injects noise without selection")
    likely_causes.append("frame-repeat VideoMAE tokens are still a temporal-representation limitation")

    recommendation = recommend_step32(best_arch, best_score, current_level, full_noise, proxy_prior_helped, delta_gain)
    return {
        "teacher_failure_analysis_performed": True,
        "teacher_topk_underperforms_proxy_confirmed": teacher_under_proxy,
        "teacher_proxy_low_correlation_confirmed": abs(float(step30b_eval.get("teacher_proxy_pearson_mean", 0.0))) < 0.1,
        "best_architecture_variant": best_arch,
        "best_diagnostic_score": best_score,
        "current_dominance_level": current_level,
        "full_context_noise_confirmed": full_noise,
        "likely_failure_causes": likely_causes,
        "h1_teacher_architecture_too_weak": bool(proxy_prior_helped or best_arch != "current_conditioned_attention_predictor"),
        "h2_occlusion_not_faithful": teacher_under_proxy and best_score == "proxy_importance",
        "h3_frame_repeat_temporal_limitation": True,
        "h4_short_horizon_current_dominance": current_level in ("high", "medium"),
        "h5_data_diversity_limited": True,
        "recommended_step32": recommendation,
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "safety_gate_pass": True,
    }


def recommend_step32(
    best_arch: str,
    best_score: str,
    current_level: str,
    full_noise: bool,
    proxy_prior_helped: bool,
    delta_gain: bool,
) -> dict[str, str]:
    if delta_gain:
        return {
            "name": "longer-horizon BridgeData window builder and target diagnosis",
            "condition": "delta target exposes much stronger context gain under existing tokens",
            "scope": "no selector/current-importance training yet",
        }
    if current_level == "high":
        return {
            "name": "longer-horizon BridgeData window builder and target diagnosis",
            "condition": "future-summary targets remain current-dominant under existing frame-repeat tokens",
            "scope": "no selector/current-importance training yet",
        }
    if proxy_prior_helped or "proxy_prior" in best_arch:
        return {
            "name": "improve teacher architecture with proxy-prior attention / temporal frame attention",
            "condition": "diagnostic architecture ablation favors proxy-prior or temporal attention variants",
            "scope": "no selector/current-importance training yet",
        }
    if full_noise and best_score == "proxy_importance":
        return {
            "name": "true temporal VideoMAE token extraction ablation",
            "condition": "selected context helps but frame-repeat/full-context signals remain noisy",
            "scope": "no selector/current-importance training yet",
        }
    return {
        "name": "add data diversity before selector training",
        "condition": "64 windows from one shard are still too small for a final context utility claim",
        "scope": "no selector/current-importance training yet",
    }
