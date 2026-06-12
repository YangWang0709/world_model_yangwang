"""Metrics for Step30B teacher-topK utility sanity."""

from __future__ import annotations

from typing import Any


def summarize_teacher_topk_utility(runs: list[dict[str, Any]]) -> dict[str, Any]:
    teacher_current = []
    teacher_random = []
    teacher_proxy = []
    rows = []
    for run in runs:
        by_policy = {str(item["policy"]): item for item in run.get("policy_metrics", [])}
        current = _loss(by_policy, "current_only")
        random = _loss(by_policy, "random_context_topk")
        proxy = _loss(by_policy, "proxy_importance_topk")
        teacher = _loss(by_policy, "teacher_occlusion_topk")
        full = _loss(by_policy, "full_context_reference")
        teacher_current.append(teacher < current)
        teacher_random.append(teacher < random)
        teacher_proxy.append(teacher < proxy)
        rows.append(
            {
                "split_seed": int(run.get("split_seed", -1)),
                "current_only": current,
                "random_context_topk": random,
                "proxy_importance_topk": proxy,
                "teacher_occlusion_topk": teacher,
                "full_context_reference": full,
            }
        )
    teacher_beats_proxy_fraction = _fraction(teacher_proxy)
    signal = "teacher_topk_stronger_than_proxy" if teacher_beats_proxy_fraction >= 0.67 else "teacher_topk_not_better_than_proxy"
    return {
        "stage": "bridgedata_v2_tfds_occlusion_teacher_step30b",
        "teacher_topk_utility_performed": bool(runs),
        "num_runs": len(runs),
        "per_seed_val_table": rows,
        "teacher_beats_current_fraction": _fraction(teacher_current),
        "teacher_beats_random_fraction": _fraction(teacher_random),
        "teacher_beats_proxy_fraction": teacher_beats_proxy_fraction,
        "teacher_mean_improvement_over_proxy": _mean_improvement(rows, "proxy_importance_topk", "teacher_occlusion_topk"),
        "teacher_topk_signal": signal,
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "safety_gate_pass": True,
    }


def recommended_step31_from_teacher_topk(summary: dict[str, Any], teacher_label_nontrivial: bool = True) -> dict[str, str]:
    if not teacher_label_nontrivial:
        return {
            "name": "diagnose teacher architecture / temporal tokens / horizon before selector training",
            "condition": "teacher occlusion labels degenerated",
            "scope": "no selector/current-importance training yet",
        }
    if float(summary.get("teacher_beats_proxy_fraction", 0.0)) >= 0.67:
        return {
            "name": "teacher-label context utility expansion before selector training",
            "condition": "teacher_topK is more stable than proxy_topK",
            "scope": "no selector/current-importance training yet",
        }
    if float(summary.get("teacher_beats_proxy_fraction", 0.0)) > 0.0:
        return {
            "name": "use teacher label as stronger target and scale windows/shards before selector training",
            "condition": "teacher_topK is close to proxy_topK but not clearly stronger",
            "scope": "no selector/current-importance training yet",
        }
    return {
        "name": "diagnose teacher architecture / temporal tokens / horizon before selector training",
        "condition": "teacher_topK does not beat proxy_topK",
        "scope": "no selector/current-importance training yet",
    }


def _loss(by_policy: dict[str, dict[str, Any]], policy: str) -> float:
    return float(by_policy.get(policy, {}).get("val_final_loss", float("inf")))


def _fraction(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(1 for flag in flags if flag) / len(flags))


def _mean_improvement(rows: list[dict[str, Any]], baseline_key: str, candidate_key: str) -> float:
    values = []
    for row in rows:
        baseline = float(row.get(baseline_key, 0.0))
        candidate = float(row.get(candidate_key, 0.0))
        if abs(baseline) > 0.0:
            values.append((baseline - candidate) / abs(baseline))
    if not values:
        return 0.0
    return float(sum(values) / len(values))
