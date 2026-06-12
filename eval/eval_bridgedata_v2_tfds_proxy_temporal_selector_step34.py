"""Evaluate Step34 proxy-supervised temporal selector planning scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_tfds_proxy_temporal_selector_step34 import (
    build_step34_selector_plan,
    load_step34_config,
    prepare_step34_proxy_temporal_selector,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_proxy_temporal_selector_step34.yaml"


def evaluate_step34_proxy_temporal_selector(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_step34_config(config_path)
    plan_path = Path(config["output"]["plan_summary_json"])
    plan = _read_json(plan_path) if plan_path.exists() else build_step34_selector_plan(config)
    safe_stop = bool(plan.get("safe_stop", False))
    decision = plan.get("decision", {})
    dry_run = plan.get("dry_run_summary", {})
    leakage = plan.get("leakage_checks", {})
    evidence = plan.get("step33b_evidence", {})
    safety_gate = bool(plan.get("safety_gate_pass", False)) and _explicit_negative_flags(decision, dry_run)
    passed = (
        not safe_stop
        and safety_gate
        and bool(decision.get("may_prepare_proxy_supervised_selector_training", False))
        and bool(dry_run.get("selector_forward_performed", False))
        and bool(dry_run.get("videomae_frozen", False))
        and bool(leakage.get("safety_gate_pass", False))
        and bool(evidence.get("proxy_signal_stable_across_shards", False))
    )
    summary = {
        "stage": config["stage"],
        "pass": bool(passed),
        "safe_stop": bool(safe_stop),
        "reason": plan.get("reason"),
        "safety_gate_pass": bool(safety_gate),
        "may_prepare_proxy_supervised_selector_training": bool(
            decision.get("may_prepare_proxy_supervised_selector_training", False)
        ),
        "selector_forward_dry_run_performed": bool(dry_run.get("selector_forward_performed", False)),
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "context_utility_claim_allowed": False,
        "final_selector_training_allowed": False,
        "selector_training_performed": False,
        "current_importance_training_performed": False,
        "videomae_training_performed": False,
        "videomae_frozen": bool(dry_run.get("videomae_frozen", True)),
        "optimizer_step_performed": False,
        "checkpoint_saved": False,
        "new_dataset_download_performed": False,
        "model_download_performed": False,
        "downstream_task_improvement_used_as_evidence": False,
        "strict_shard_aware_splits": bool((plan.get("split_plan") or {}).get("strict_shard_aware_splits", False)),
        "no_language_or_trajectory_leakage": bool(leakage.get("no_language_or_trajectory_leakage", False)),
        "dataset_bias_detected": bool(evidence.get("dataset_bias_detected", False)),
        "distribution_shift_flags": evidence.get("distribution_shift_flags", {}),
        "full_context_noise_acknowledged": bool(evidence.get("full_context_noise_confirmed", False)),
        "future_selector_training_gate_ready": False,
        "decision_flags": {
            "selector_training_allowed": False,
            "current_importance_training_allowed": False,
            "context_utility_claim_allowed": False,
            "final_selector_training_allowed": False,
        },
    }
    _write_json(config["output"]["eval_json"], summary)
    Path(config["output"]["eval_md"]).write_text(_eval_markdown(summary), encoding="utf-8")
    _write_step34_docs(config, plan, summary)
    return summary


def _explicit_negative_flags(decision: dict[str, Any], dry_run: dict[str, Any]) -> bool:
    return (
        not bool(decision.get("context_utility_claim_allowed", False))
        and not bool(decision.get("selector_training_allowed", False))
        and not bool(decision.get("current_importance_training_allowed", False))
        and not bool(decision.get("final_selector_training_allowed", False))
        and not bool(dry_run.get("selector_training_performed", False))
        and not bool(dry_run.get("current_importance_training_performed", False))
        and not bool(dry_run.get("optimizer_step_performed", False))
        and not bool(dry_run.get("checkpoint_saved", False))
    )


def _write_step34_docs(config: dict[str, Any], plan: dict[str, Any], summary: dict[str, Any]) -> None:
    plan_doc = Path(config["output"]["plan_doc_md"])
    gates_doc = Path(config["output"]["gates_doc_md"])
    plan_doc.parent.mkdir(parents=True, exist_ok=True)
    gates_doc.parent.mkdir(parents=True, exist_ok=True)
    decision = plan.get("decision", {})
    evidence = plan.get("step33b_evidence", {})
    dry_run = plan.get("dry_run_summary", {})
    plan_doc.write_text(
        "\n".join(
            [
                "# Step34 Proxy-Supervised Temporal Selector Plan",
                "",
                "Step34 converts the stable Step33B proxy temporal signal into a safe selector-training scaffold.",
                "",
                "Allowed:",
                "- This step may prepare proxy-supervised selector training.",
                "- This step may instantiate a lightweight temporal selector head for dry-run forward checks.",
                "- This step may read existing local Step33B token and proxy-importance artifacts.",
                "- This step must use strict shard-aware splits.",
                "",
                "Not allowed:",
                "- This step must not claim context utility.",
                "- This step must not train final selector/current importance.",
                "- This step must not use downstream task improvement as evidence.",
                "- This step must not download extra datasets or models.",
                "- This step must keep VideoMAE frozen.",
                "- This step must not save selector checkpoints.",
                "",
                "Step33B evidence used for planning:",
                f"- proxy_signal_stable_across_shards: `{str(evidence.get('proxy_signal_stable_across_shards')).lower()}`",
                f"- within proxy gain/current: `{evidence.get('shard1_within_proxy_gain_over_current_mean')}`",
                f"- cross proxy gain/current: `{evidence.get('cross_proxy_gain_over_current_mean')}`",
                f"- mixed proxy gain/current: `{evidence.get('mixed_proxy_gain_over_current_mean')}`",
                f"- dataset_bias_detected: `{str(evidence.get('dataset_bias_detected')).lower()}`",
                f"- full_context_noise_confirmed: `{str(evidence.get('full_context_noise_confirmed')).lower()}`",
                "",
                "Step34 decision:",
                f"- may_prepare_proxy_supervised_selector_training: `{str(decision.get('may_prepare_proxy_supervised_selector_training')).lower()}`",
                f"- selector_forward_dry_run_performed: `{str(dry_run.get('selector_forward_performed')).lower()}`",
                "- selector_training_allowed: `false`",
                "- current_importance_training_allowed: `false`",
                "- context_utility_claim_allowed: `false`",
                "- future_selector_training_gate_ready: `false`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    gates = plan.get("future_training_gates", {})
    gates_doc.write_text(
        "\n".join(
            [
                "# BridgeData V2 Step34 Selector Training Gates",
                "",
                "Future selector training can only be considered after all gates below are satisfied in a bounded follow-up step.",
                "",
                f"- selector beats random baseline: `{gates.get('selector_beats_random_baseline', {}).get('status')}`",
                f"- selector beats current-only baseline: `{gates.get('selector_beats_current_only_baseline', {}).get('status')}`",
                f"- selector generalizes across shard: `{gates.get('selector_generalizes_across_shard', {}).get('status')}`",
                f"- no language/trajectory leakage: `{gates.get('no_language_or_trajectory_leakage', {}).get('status')}`",
                f"- no dataset bias or shard shift: `{gates.get('no_dataset_bias_or_shard_shift', {}).get('status')}`",
                f"- full-context-noisy issue acknowledged: `{gates.get('full_context_noisy_issue_acknowledged', {}).get('status')}`",
                "",
                "Required evaluation design:",
                "- within-shard validation: shard1 train/validation split with disjoint sample IDs and trajectories where possible.",
                "- cross-shard validation: train shard0 -> validate shard1 and train shard1 -> validate shard0.",
                "- mixed-shard validation: shard-aware mixed training and validation with per-shard breakdown.",
                "",
                "Decision flags remain conservative:",
                f"- selector_training_allowed: `{str(summary.get('selector_training_allowed')).lower()}`",
                f"- current_importance_training_allowed: `{str(summary.get('current_importance_training_allowed')).lower()}`",
                f"- context_utility_claim_allowed: `{str(summary.get('context_utility_claim_allowed')).lower()}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _eval_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step34 Eval",
            "",
            f"- pass: `{str(summary.get('pass')).lower()}`",
            f"- safe_stop: `{str(summary.get('safe_stop')).lower()}`",
            f"- safety_gate_pass: `{str(summary.get('safety_gate_pass')).lower()}`",
            f"- selector_training_allowed: `{str(summary.get('selector_training_allowed')).lower()}`",
            f"- current_importance_training_allowed: `{str(summary.get('current_importance_training_allowed')).lower()}`",
            f"- context_utility_claim_allowed: `{str(summary.get('context_utility_claim_allowed')).lower()}`",
        ]
    ) + "\n"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare_step34_proxy_temporal_selector(args.config)
    print(json.dumps(evaluate_step34_proxy_temporal_selector(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
