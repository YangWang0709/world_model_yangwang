"""Run Step26 BridgeData TFDS context bottleneck world-model smoke."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_tfds_context_selection import select_context_tokens, summarize_selection
from data.bridgedata_v2_tfds_world_model_batch import (
    load_bridge_tfds_world_model_samples,
    step24_step25_inputs_missing,
    summarize_world_model_samples,
)
from data.bridgedata_v2_tfds_world_model_smoke_metrics import (
    LOSS_QUALITY_NOTE,
    policy_metric_table,
    summarize_forward_loss,
    summarize_policy_metrics,
)
from models.bridgedata_v2_context_bottleneck_smoke_model import (
    BridgeDataContextBottleneckSmokePredictor,
    forward_loss_for_policy,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_world_model_smoke_step26.yaml"


def run_bridge_tfds_world_model_smoke(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(config_path)
    output = config["output"]
    paths = _output_paths(output)

    env_guard = _env_isaaclab_guard()
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        summary = _safe_stop_summary(config, "env_isaaclab is polluted with TensorFlow/TFDS", env_guard)
        _write_all_outputs(paths, summary, {}, {"policy_metrics": []}, {"losses": []})
        return summary

    missing = step24_step25_inputs_missing(config)
    if missing:
        summary = _safe_stop_summary(config, f"missing Step24/25 artifacts: {missing}", env_guard)
        _write_all_outputs(paths, summary, {}, {"policy_metrics": []}, {"losses": []})
        return summary

    samples = load_bridge_tfds_world_model_samples(config)
    batch_summary = summarize_world_model_samples(samples)
    hidden_dim = int(config["world_model_smoke"].get("hidden_dim", 512))
    model = BridgeDataContextBottleneckSmokePredictor(hidden_dim=hidden_dim, seed=int(config.get("seed", 42)))
    model.eval()

    forward_records: list[dict[str, Any]] = []
    selection_summaries: list[dict[str, Any]] = []
    for sample in samples:
        for policy in config["policies"]:
            selection = select_context_tokens(sample, policy)
            selection_summaries.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    **summarize_selection(selection),
                }
            )
            forward_records.append(forward_loss_for_policy(sample, selection, model))

    policy_metrics = summarize_policy_metrics(forward_records)
    forward_summary = summarize_forward_loss(forward_records)
    policies_evaluated = [str(policy["name"]) for policy in config["policies"]]
    summary = {
        "stage": config["stage"],
        "world_model_smoke_performed": bool(samples) and bool(forward_records),
        "safe_stop": not (bool(samples) and bool(forward_records)),
        "reason": None if samples and forward_records else "No Step26 forward records were generated.",
        "num_samples": len(samples),
        "policies_evaluated": policies_evaluated,
        "topk_values": list(config["context_bottleneck"].get("topk_values", [])),
        "token_shapes": batch_summary.get("token_shapes", {}),
        "importance_shape": batch_summary.get("importance_shape"),
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "optimizer_step_performed": False,
        "training_performed": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "world_model_training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "download_performed": False,
        "model_download_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "large_token_shards_generated": False,
        "large_importance_shards_generated": False,
        "current_importance_generated": False,
        "action_used_as_input": False,
        "language_used_as_input": False,
        "goal_used_as_input": False,
        "all_losses_finite": bool(forward_summary.get("all_losses_finite", False)),
        "random_init_result_not_scientific": True,
        "loss_quality_note": LOSS_QUALITY_NOTE,
        "policy_metric_table": policy_metric_table(policy_metrics),
        "selection_summaries": selection_summaries,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": True,
    }
    policy_payload = {
        "stage": config["stage"],
        "policy_metrics": policy_metrics,
        "selection_summaries": selection_summaries,
        "loss_quality_note": LOSS_QUALITY_NOTE,
    }
    _write_all_outputs(paths, summary, batch_summary, policy_payload, forward_summary)
    return summary


def _output_paths(output: dict[str, Any]) -> dict[str, Path]:
    return {key: Path(value) for key, value in output.items() if key.endswith("_json") or key.endswith("_md")}


def _safe_stop_summary(config: dict[str, Any], reason: str, env_guard: dict[str, bool]) -> dict[str, Any]:
    return {
        "stage": config["stage"],
        "world_model_smoke_performed": False,
        "safe_stop": True,
        "reason": reason,
        "num_samples": 0,
        "policies_evaluated": [str(policy["name"]) for policy in config.get("policies", [])],
        "topk_values": list(config.get("context_bottleneck", {}).get("topk_values", [])),
        "token_shapes": {},
        "importance_shape": None,
        "current_tokens_kept_full": True,
        "train_current_importance": False,
        "optimizer_step_performed": False,
        "training_performed": False,
        "teacher_training_performed": False,
        "selector_training_performed": False,
        "world_model_training_performed": False,
        "token_extraction_performed": False,
        "importance_generation_performed": False,
        "download_performed": False,
        "model_download_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "large_token_shards_generated": False,
        "large_importance_shards_generated": False,
        "current_importance_generated": False,
        "all_losses_finite": False,
        "random_init_result_not_scientific": True,
        "loss_quality_note": LOSS_QUALITY_NOTE,
        "policy_metric_table": [],
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": not (env_guard["tensorflow"] or env_guard["tensorflow_datasets"]),
    }


def _write_all_outputs(
    paths: dict[str, Path],
    summary: dict[str, Any],
    batch_summary: dict[str, Any],
    policy_payload: dict[str, Any],
    forward_summary: dict[str, Any],
) -> None:
    _write_json(paths["batch_summary_json"], batch_summary)
    _write_json(paths["policy_metrics_json"], policy_payload)
    _write_json(paths["forward_loss_json"], forward_summary)
    _write_json(paths["smoke_summary_json"], summary)
    paths["smoke_summary_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["smoke_summary_md"].write_text(_format_summary_md(summary), encoding="utf-8")


def _format_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# BridgeData V2 TFDS World-Model Smoke Summary",
        "",
        f"- world_model_smoke_performed: `{str(summary.get('world_model_smoke_performed', False)).lower()}`",
        f"- safe_stop: `{str(summary.get('safe_stop', False)).lower()}`",
        f"- num_samples: `{summary.get('num_samples')}`",
        f"- policies_evaluated: `{summary.get('policies_evaluated')}`",
        f"- all_losses_finite: `{str(summary.get('all_losses_finite', False)).lower()}`",
        f"- current_tokens_kept_full: `{str(summary.get('current_tokens_kept_full', False)).lower()}`",
        f"- train_current_importance: `{str(summary.get('train_current_importance', True)).lower()}`",
        f"- optimizer_step_performed: `{str(summary.get('optimizer_step_performed', True)).lower()}`",
        f"- training_performed: `{str(summary.get('training_performed', True)).lower()}`",
        f"- token_extraction_performed: `{str(summary.get('token_extraction_performed', True)).lower()}`",
        f"- importance_generation_performed: `{str(summary.get('importance_generation_performed', True)).lower()}`",
        f"- random_init_result_not_scientific: `{str(summary.get('random_init_result_not_scientific', False)).lower()}`",
        f"- loss_quality_note: `{summary.get('loss_quality_note')}`",
    ]
    if summary.get("reason"):
        lines.append(f"- reason: `{summary['reason']}`")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _env_isaaclab_guard() -> dict[str, bool]:
    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run_bridge_tfds_world_model_smoke(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
