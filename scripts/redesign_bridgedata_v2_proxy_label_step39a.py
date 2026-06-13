"""Run Step39A proxy-label redesign diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_proxy_label_redesign_step39a import (
    STAGE,
    build_safe_stop_payload,
    build_step39a_gate_decision,
    load_step39a_config,
    missing_step39a_inputs,
    run_step39a_redesign_statistics,
    write_step39a_json,
)
from eval.eval_bridgedata_v2_proxy_label_redesign_step39a import evaluate_step39a_proxy_label_redesign

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_proxy_label_redesign_step39a.yaml"


def run_step39a_proxy_label_redesign(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_step39a_config(config_path)
    _assert_safety_guards(config)
    missing = missing_step39a_inputs(config)
    if missing:
        summary = build_safe_stop_payload(config, missing)
        metrics = {"stage": STAGE, "safe_stop": True, "num_rows": 0, "rows": [], "aggregates": {}}
        comparison = {"stage": STAGE, "safe_stop": True, "ranking": {"ranked_variants": [], "best_variant": None}}
        gate = build_step39a_gate_decision(comparison=comparison, step38b_gate={})
        _write_outputs(config, summary, metrics, comparison, gate)
        eval_summary = evaluate_step39a_proxy_label_redesign(config_path)
        return {"summary": summary, "comparison": comparison, "gate_decision": gate, "eval": eval_summary}

    payload = run_step39a_redesign_statistics(config)
    step38b_gate = json.loads(Path(config["input"]["step38b_gate_decision_json"]).read_text(encoding="utf-8"))
    gate = build_step39a_gate_decision(comparison=payload["comparison"], step38b_gate=step38b_gate)
    _write_outputs(config, payload["summary"], payload["metrics"], payload["comparison"], gate)
    eval_summary = evaluate_step39a_proxy_label_redesign(config_path)
    return {"summary": payload["summary"], "comparison": payload["comparison"], "gate_decision": gate, "eval": eval_summary}


def _assert_safety_guards(config: dict[str, Any]) -> None:
    guards = config.get("guards", {})
    required_true = [
        "no_new_data_download",
        "no_new_model_download",
        "no_training",
        "no_optimizer_step",
        "no_checkpoint_save",
        "no_state_dict_save",
        "no_tensorflow_import_in_env_isaaclab",
        "no_tensorflow_datasets_import_in_env_isaaclab",
        "no_write_data_token_shards",
        "no_write_data_importance_shards",
    ]
    bad = [flag for flag in required_true if not bool(guards.get(flag, False))]
    if bad:
        raise ValueError(f"unsafe Step39A guard configuration: {bad}")
    data_cfg = config.get("data", {})
    if bool(data_cfg.get("use_action_as_input", True)):
        raise ValueError("Step39A must not use action as input")
    if bool(data_cfg.get("use_language_as_input", True)):
        raise ValueError("Step39A must not use language as input")
    if bool(data_cfg.get("load_future_tokens_for_label_redesign", True)):
        raise ValueError("Step39A must not load future tokens")


def _write_outputs(
    config: dict[str, Any],
    summary: dict[str, Any],
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    gate: dict[str, Any],
) -> None:
    write_step39a_json(config["output"]["label_redesign_summary_json"], summary)
    write_step39a_json(config["output"]["label_variant_metrics_json"], metrics)
    write_step39a_json(config["output"]["label_variant_comparison_json"], comparison)
    write_step39a_json(config["output"]["gate_decision_json"], gate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = run_step39a_proxy_label_redesign(args.config)
    print(json.dumps(result["eval"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
