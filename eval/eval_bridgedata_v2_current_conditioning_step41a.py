"""Evaluate and report Step41A current-conditioning diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_current_conditioning_dataset_step41a import OPTIMIZER_SCOPE, STAGE, load_step41a_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_current_conditioning_diagnosis_step41a.yaml"


def evaluate_step41a_current_conditioning(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_step41a_config(config_path)
    train_summary = _read_json(config["output"]["train_summary_json"])
    loss_curves = _read_json(config["output"]["loss_curves_json"])
    metrics = _read_json(config["output"]["current_conditioning_metrics_json"])
    variant_comparison = _read_json(config["output"]["variant_comparison_json"])
    gate = _read_json(config["output"]["gate_decision_json"])
    safety_gate = _safety_gate(train_summary, metrics, gate)
    best_variant = gate.get("best_variant")
    no_current = _variant_overall(variant_comparison, "no_current_context_only")
    current_mean = _variant_overall(variant_comparison, "current_mean_summary")
    best_within = _variant_eval(variant_comparison, best_variant, "within_shard")
    best_cross = _variant_eval(variant_comparison, best_variant, "cross_shard")
    best_mixed = _variant_eval(variant_comparison, best_variant, "mixed_shard")
    result = {
        "stage": STAGE,
        "pass": bool(not train_summary.get("safe_stop", False) and safety_gate and train_summary.get("training_performed", False)),
        "safe_stop": bool(train_summary.get("safe_stop", False)),
        "reason": train_summary.get("reason"),
        "label_variant": train_summary.get("label_variant"),
        "variants_run": train_summary.get("variants_run", []),
        "best_variant": best_variant,
        "no_current_mse": no_current.get("selector_val_mse"),
        "current_mean_summary_mse": current_mean.get("selector_val_mse"),
        "best_variant_within_mse": best_within.get("selector_val_mse"),
        "best_variant_cross_mse": best_cross.get("selector_val_mse"),
        "best_variant_mixed_mse": best_mixed.get("selector_val_mse"),
        "bounded_current_conditioning_smoke_training_performed": bool(
            train_summary.get("bounded_current_conditioning_smoke_training_performed", False)
        ),
        "optimizer_step_performed": bool(train_summary.get("optimizer_step_performed", False)),
        "optimizer_scope": train_summary.get("optimizer_scope"),
        "best_variant_beats_no_current": bool(gate.get("best_variant_beats_no_current", False)),
        "best_variant_beats_current_mean_summary": bool(gate.get("best_variant_beats_current_mean_summary", False)),
        "best_variant_beats_random_token_baseline": bool(gate.get("best_variant_beats_random_token_baseline", False)),
        "best_variant_beats_uniform_or_mean_baseline": bool(
            gate.get("best_variant_beats_uniform_or_mean_baseline", False)
        ),
        "best_variant_beats_temporal_broadcast_baseline": bool(
            gate.get("best_variant_beats_temporal_broadcast_baseline", False)
        ),
        "best_variant_beats_train_global_spatial_prior_baseline": bool(
            gate.get("best_variant_beats_train_global_spatial_prior_baseline", False)
        ),
        "best_variant_generalizes_cross_shard": bool(gate.get("best_variant_generalizes_cross_shard", False)),
        "top256_overlap_mean": float(gate.get("top256_overlap_mean", 0.0)),
        "overfit_gap_mean_abs": float(gate.get("overfit_gap_mean_abs", 0.0)),
        "current_conditioning_candidate_ready": bool(gate.get("current_conditioning_candidate_ready", False)),
        "recommended_step42": gate.get("recommended_step42", {}),
        "downstream_selector_use_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "context_utility_claim_allowed": False,
        "final_selector_training_performed": False,
        "current_importance_training_performed": False,
        "world_model_training_performed": False,
        "downstream_task_training_performed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
        "new_dataset_download_performed": False,
        "model_download_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "future_tokens_exposed_to_selector": False,
        "future_tokens_used_as_input": False,
        "tensorflow_required": False,
        "tensorflow_datasets_required": False,
        "extra_cloud_resources_used": False,
        "safety_gate_pass": bool(safety_gate),
    }
    _write_json(config["output"]["eval_json"], result)
    Path(config["output"]["eval_md"]).write_text(_eval_markdown(result), encoding="utf-8")
    _write_docs(train_summary, loss_curves, metrics, variant_comparison, gate, result)
    return result


def _safety_gate(train_summary: dict[str, Any], metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        bool(train_summary.get("bounded_current_conditioning_smoke_training_performed", False))
        and bool(train_summary.get("optimizer_step_performed", False))
        and train_summary.get("optimizer_scope") == OPTIMIZER_SCOPE
        and not bool(train_summary.get("videomae_loaded", False))
        and not bool(train_summary.get("videomae_training_performed", False))
        and not bool(train_summary.get("current_importance_training_performed", False))
        and not bool(train_summary.get("final_selector_training_performed", False))
        and not bool(train_summary.get("world_model_training_performed", False))
        and not bool(train_summary.get("downstream_task_training_performed", False))
        and not bool(train_summary.get("checkpoint_saved", False))
        and not bool(train_summary.get("state_dict_saved", False))
        and not bool(train_summary.get("data_token_shards_written", False))
        and not bool(train_summary.get("data_importance_shards_written", False))
        and not bool(train_summary.get("action_used_as_input", False))
        and not bool(train_summary.get("language_used_as_input", False))
        and not bool(train_summary.get("future_tokens_exposed_to_selector", False))
        and not bool(train_summary.get("future_tokens_used_as_input", False))
        and not bool(gate.get("downstream_selector_use_allowed", False))
        and not bool(gate.get("final_selector_training_allowed", False))
        and not bool(gate.get("current_importance_training_allowed", False))
        and not bool(gate.get("context_utility_claim_allowed", False))
        and bool(metrics.get("safety_gate_pass", False))
    )


def _write_docs(
    train_summary: dict[str, Any],
    loss_curves: dict[str, Any],
    metrics: dict[str, Any],
    variant_comparison: dict[str, Any],
    gate: dict[str, Any],
    result: dict[str, Any],
) -> None:
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "STEP41A_CURRENT_CONDITIONING_DIAGNOSIS.md").write_text(
        "\n".join(
            [
                "# Step41A Current-Conditioning Diagnosis",
                "",
                "Step41A is a bounded diagnostic of how the student selector looks at the current observation while choosing historical context tokens.",
                "It is not final selector training, not downstream training, not current importance training, and not evidence of final context utility.",
                "",
                "Allowed:",
                "- prepare and run bounded current-conditioning selector smoke training",
                "- update only the lightweight current-conditioned selector head",
                "- keep VideoMAE frozen and unloaded",
                "- use strict shard-aware within, cross, and mixed validation splits",
                "- reuse existing local token and importance artifacts only",
                "",
                "Not allowed:",
                "- claim context utility",
                "- train final selector, current importance, world model, VLM, RL, or downstream tasks",
                "- use downstream task improvement as evidence",
                "- download extra datasets, models, images, videos, checkpoints, or archives",
                "- save checkpoint or state_dict artifacts",
                "",
                f"- variants_run: `{json.dumps(result.get('variants_run'), sort_keys=True)}`",
                f"- best_variant: `{result.get('best_variant')}`",
                f"- optimizer_scope: `{result.get('optimizer_scope')}`",
                f"- num_curves: `{loss_curves.get('num_curves')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (docs_dir / "BRIDGEDATA_V2_CURRENT_CONDITIONING_VARIANT_REPORT.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Current-Conditioning Variant Report",
                "",
                "All metrics in this report are recomputed against the Step41A redesigned target label.",
                "Step40A metrics are background only and are not reused as this round's result.",
                "",
                f"- no_current_mse: `{result.get('no_current_mse')}`",
                f"- current_mean_summary_mse: `{result.get('current_mean_summary_mse')}`",
                f"- best_variant_within_mse: `{result.get('best_variant_within_mse')}`",
                f"- best_variant_cross_mse: `{result.get('best_variant_cross_mse')}`",
                f"- best_variant_mixed_mse: `{result.get('best_variant_mixed_mse')}`",
                f"- best_variant_beats_no_current: `{str(result.get('best_variant_beats_no_current')).lower()}`",
                f"- best_variant_beats_current_mean_summary: `{str(result.get('best_variant_beats_current_mean_summary')).lower()}`",
                f"- beats_random: `{str(result.get('best_variant_beats_random_token_baseline')).lower()}`",
                f"- beats_uniform_or_mean: `{str(result.get('best_variant_beats_uniform_or_mean_baseline')).lower()}`",
                f"- beats_temporal_broadcast: `{str(result.get('best_variant_beats_temporal_broadcast_baseline')).lower()}`",
                f"- beats_train_global_spatial_prior: `{str(result.get('best_variant_beats_train_global_spatial_prior_baseline')).lower()}`",
                f"- cross_shard_generalization: `{str(result.get('best_variant_generalizes_cross_shard')).lower()}`",
                f"- top256_overlap_mean: `{result.get('top256_overlap_mean')}`",
                f"- overfit_gap_mean_abs: `{result.get('overfit_gap_mean_abs')}`",
                "",
                "The coarse spatial query variant uses contiguous token-index bins. This is a low-cost diagnostic heuristic, not true pixel geometry.",
                "",
                f"Ranked variants: `{json.dumps(variant_comparison.get('ranked_variants', []), sort_keys=True)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    recommended = gate.get("recommended_step42", {})
    (docs_dir / "BRIDGEDATA_V2_STEP41A_DECISION.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Step41A Decision",
                "",
                "Step40A failed because it beat random and train-global-spatial-prior baselines but did not beat uniform/mean or temporal-broadcast baselines, and cross-shard generalization was false.",
                "Step41A therefore asks whether stronger current-conditioning actually helps the selector choose history, rather than treating this as another final selector attempt.",
                "",
                f"- current_conditioning_candidate_ready: `{str(gate.get('current_conditioning_candidate_ready')).lower()}`",
                f"- best_variant: `{gate.get('best_variant')}`",
                f"- best_variant_beats_no_current: `{str(gate.get('best_variant_beats_no_current')).lower()}`",
                f"- best_variant_beats_current_mean_summary: `{str(gate.get('best_variant_beats_current_mean_summary')).lower()}`",
                f"- best_variant_generalizes_cross_shard: `{str(gate.get('best_variant_generalizes_cross_shard')).lower()}`",
                f"- reason: `{gate.get('reason')}`",
                "",
                "Recommended Step42:",
                f"- name: `{recommended.get('name')}`",
                f"- scope: `{recommended.get('scope')}`",
                "",
                "Safety decision flags:",
                "- downstream_selector_use_allowed: `false`",
                "- final_selector_training_allowed: `false`",
                "- current_importance_training_allowed: `false`",
                "- context_utility_claim_allowed: `false`",
                "- checkpoint_saved: `false`",
                "- state_dict_saved: `false`",
                "- new_dataset_download_performed: `false`",
                "- model_download_performed: `false`",
                "- action/language/future tokens used as selector input: `false`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _eval_markdown(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step41A Eval",
            "",
            f"- pass: `{str(result.get('pass')).lower()}`",
            f"- safe_stop: `{str(result.get('safe_stop')).lower()}`",
            f"- safety_gate_pass: `{str(result.get('safety_gate_pass')).lower()}`",
            f"- best_variant: `{result.get('best_variant')}`",
            f"- current_conditioning_candidate_ready: `{str(result.get('current_conditioning_candidate_ready')).lower()}`",
            f"- recommended_step42: `{(result.get('recommended_step42') or {}).get('name')}`",
            "- downstream_selector_use_allowed: `false`",
            "- final_selector_training_allowed: `false`",
            "- current_importance_training_allowed: `false`",
            "- context_utility_claim_allowed: `false`",
        ]
    ) + "\n"


def _variant_overall(variant_comparison: dict[str, Any], variant: str | None) -> dict[str, Any]:
    if not variant:
        return {}
    return variant_comparison.get("per_variant", {}).get(variant, {}).get("overall", {})


def _variant_eval(variant_comparison: dict[str, Any], variant: str | None, eval_key: str) -> dict[str, Any]:
    if not variant:
        return {}
    return variant_comparison.get("per_variant", {}).get(variant, {}).get(eval_key, {})


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(evaluate_step41a_current_conditioning(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
