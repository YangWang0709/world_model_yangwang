"""Run Step38B proxy label spatial-detail learnability diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_proxy_label_diagnosis_step38b import (
    STAGE,
    build_cross_shard_consistency,
    build_safe_stop_payload,
    build_step38b_diagnosis,
    build_step38b_gate_decision,
    diagnose_step38b_sample,
    load_step38b_config,
    load_step38b_importance_samples,
    missing_step38b_inputs,
    write_step38b_json,
)
from data.bridgedata_v2_proxy_patch_selector_splits_step36 import summarize_step36_leakage
from data.bridgedata_v2_proxy_label_diagnosis_step38b import build_step38b_split_settings
from eval.eval_bridgedata_v2_proxy_label_diagnosis_step38b import evaluate_step38b_proxy_label_diagnosis

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_proxy_label_diagnosis_step38b.yaml"


def run_step38b_proxy_label_diagnosis(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_step38b_config(config_path)
    _assert_safety_guards(config)
    missing = missing_step38b_inputs(config)
    if missing:
        summary = build_safe_stop_payload(config, missing)
        label_decomposition = {"stage": STAGE, "num_rows": 0, "rows": [], "safe_stop": True}
        cross_shard = {
            "stage": STAGE,
            "computed": False,
            "safe_stop": True,
            "reason": "missing required local Step38B inputs",
            "cross_shard_residual_consistent": False,
        }
        gate = build_step38b_gate_decision(
            config=config,
            diagnosis_summary=summary,
            cross_shard_consistency=cross_shard,
            leakage_summary={"no_language_or_trajectory_leakage": False},
        )
        _write_outputs(config, summary, label_decomposition, cross_shard, gate)
        eval_summary = evaluate_step38b_proxy_label_diagnosis(config_path)
        return {"diagnosis_summary": summary, "gate_decision": gate, "eval": eval_summary}

    topk_values = [int(value) for value in config.get("diagnostics", {}).get("topk_values", [64, 128, 256, 512])]
    samples = load_step38b_importance_samples(config)
    rows = [diagnose_step38b_sample(sample, topk_values) for sample in samples]
    summary = build_step38b_diagnosis(config)
    label_decomposition = {
        "stage": STAGE,
        "num_rows": len(rows),
        "rows": rows,
        "schema": {
            "importance": "[16,392]",
            "temporal_component": "[16]",
            "temporal_broadcast": "[16,392]",
            "spatial_residual": "[16,392]",
        },
        "training_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
    }
    cross_shard = build_cross_shard_consistency(samples)
    leakage = summarize_step36_leakage(build_step38b_split_settings(config))
    gate = build_step38b_gate_decision(
        config=config,
        diagnosis_summary=summary,
        cross_shard_consistency=cross_shard,
        leakage_summary=leakage,
    )
    _write_outputs(config, summary, label_decomposition, cross_shard, gate)
    eval_summary = evaluate_step38b_proxy_label_diagnosis(config_path)
    return {"diagnosis_summary": summary, "gate_decision": gate, "eval": eval_summary}


def _assert_safety_guards(config: dict[str, Any]) -> None:
    guards = config.get("guards", {})
    required_true = [
        "no_new_data_download",
        "no_new_model_download",
        "no_training",
        "no_optimizer_step",
        "no_checkpoint_save",
        "no_tensorflow_import_in_env_isaaclab",
        "no_tensorflow_datasets_import_in_env_isaaclab",
        "no_write_data_token_shards",
        "no_write_data_importance_shards",
    ]
    missing = [flag for flag in required_true if not bool(guards.get(flag, False))]
    if missing:
        raise ValueError(f"unsafe Step38B guard configuration: {missing}")
    data_cfg = config.get("data", {})
    if bool(data_cfg.get("use_action_as_input", True)) or bool(data_cfg.get("use_language_as_input", True)):
        raise ValueError("Step38B must not use action or language as input")
    if bool(data_cfg.get("load_future_tokens_for_diagnosis", True)):
        raise ValueError("Step38B must not load future tokens")


def _write_outputs(
    config: dict[str, Any],
    summary: dict[str, Any],
    label_decomposition: dict[str, Any],
    cross_shard: dict[str, Any],
    gate: dict[str, Any],
) -> None:
    write_step38b_json(config["output"]["diagnosis_summary_json"], summary)
    write_step38b_json(config["output"]["label_decomposition_json"], label_decomposition)
    write_step38b_json(config["output"]["cross_shard_consistency_json"], cross_shard)
    write_step38b_json(config["output"]["gate_decision_json"], gate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = run_step38b_proxy_label_diagnosis(args.config)
    print(json.dumps(result["eval"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
