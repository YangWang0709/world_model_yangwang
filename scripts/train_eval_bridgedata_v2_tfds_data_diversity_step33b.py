"""Train/evaluate Step33B shard-diversity tiny diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_tfds_cross_shard_metrics import (
    aggregate_eval_rows,
    decide_cross_shard_stability,
    skipped_eval_summary,
)
from data.bridgedata_v2_tfds_data_diversity_proxy_importance import summarize_step33b_proxy_importance
from data.bridgedata_v2_tfds_dataset_bias_metrics import (
    shard0_summary_from_step32,
    summarize_dataset_bias,
)
from data.bridgedata_v2_tfds_longer_horizon_manifest import read_jsonl
from data.bridgedata_v2_tfds_longer_horizon_metrics import target_summary_for_variant
from data.bridgedata_v2_tfds_world_model_batch import (
    load_bridge_tfds_world_model_samples,
    validate_world_model_sample,
)
from models.bridgedata_v2_context_bottleneck_smoke_model import BridgeDataContextBottleneckSmokePredictor
from scripts.train_eval_bridgedata_v2_tfds_context_utility import OPTIMIZER_SCOPE, _optimizer_scope, _resolve_device
from scripts.train_eval_bridgedata_v2_tfds_longer_horizon_step32 import _policy_dataset, _train_policy

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_data_diversity_step33b.yaml"


def train_eval_step33b_data_diversity(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = _load_yaml(config_path)
    env_guard = _env_guard()
    if env_guard["tensorflow"] or env_guard["tensorflow_datasets"]:
        payload = _safe_stop_payload(config, "env_isaaclab is polluted with TensorFlow/TFDS", env_guard)
        _write_outputs(config, payload)
        return payload
    missing = _missing_inputs(config)
    if missing:
        payload = _safe_stop_payload(config, f"missing Step33B trainval inputs: {missing}", env_guard)
        _write_outputs(config, payload)
        return payload

    shard1_samples = _load_shard1_samples(config)
    shard0_samples, shard0_skip_reason = _load_shard0_gap0_samples(config)
    split_payload = json.loads(Path(config["output"]["shard_split_json"]).read_text(encoding="utf-8"))
    shard0_splits = _load_shard0_gap0_splits(config)
    target_variant = str(config["evaluation"].get("target_variant", "future_delta_last_minus_current"))
    train_cfg = config["tiny_trainval"]
    device = _resolve_device(str(train_cfg.get("device", "cpu")), bool(train_cfg.get("allow_cpu_fallback", True)))
    policies = [dict(policy) for policy in config["policies"] if bool(policy.get("train", True))]

    shard1_by_id = {str(sample["sample_id"]): sample for sample in shard1_samples}
    shard0_by_id = {str(sample["sample_id"]): sample for sample in shard0_samples}

    within_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    mixed_rows: list[dict[str, Any]] = []
    optimizer_step_performed = False

    for split in split_payload.get("splits", []):
        seed = int(split["split_seed"])
        shard1_train = _samples_by_ids(shard1_by_id, split["train_sample_ids"])
        shard1_val = _samples_by_ids(shard1_by_id, split["val_sample_ids"])
        row, performed = _run_policy_setting(
            config=config,
            train_cfg=train_cfg,
            policies=policies,
            train_samples=shard1_train,
            val_samples=shard1_val,
            split_seed=seed,
            setting="shard1_within",
            train_shard="shard1",
            val_shard="shard1",
            target_variant=target_variant,
            device=device,
        )
        within_rows.append(row)
        optimizer_step_performed = optimizer_step_performed or performed

        shard0_split = shard0_splits.get(seed)
        if shard0_split and shard0_by_id:
            shard0_train = _samples_by_ids(shard0_by_id, shard0_split["train_sample_ids"])
            shard0_val = _samples_by_ids(shard0_by_id, shard0_split["val_sample_ids"])
            if shard0_train and shard0_val:
                row, performed = _run_policy_setting(
                    config=config,
                    train_cfg=train_cfg,
                    policies=policies,
                    train_samples=shard0_train,
                    val_samples=shard1_val,
                    split_seed=seed,
                    setting="train_shard0_val_shard1",
                    train_shard="shard0",
                    val_shard="shard1",
                    target_variant=target_variant,
                    device=device,
                )
                cross_rows.append(row)
                optimizer_step_performed = optimizer_step_performed or performed
                row, performed = _run_policy_setting(
                    config=config,
                    train_cfg=train_cfg,
                    policies=policies,
                    train_samples=shard1_train,
                    val_samples=shard0_val,
                    split_seed=seed,
                    setting="train_shard1_val_shard0",
                    train_shard="shard1",
                    val_shard="shard0",
                    target_variant=target_variant,
                    device=device,
                )
                cross_rows.append(row)
                optimizer_step_performed = optimizer_step_performed or performed
                row, performed = _run_policy_setting(
                    config=config,
                    train_cfg=train_cfg,
                    policies=policies,
                    train_samples=shard0_train + shard1_train,
                    val_samples=shard0_val + shard1_val,
                    split_seed=seed,
                    setting="mixed_train_mixed_val_shard_aware",
                    train_shard="mixed",
                    val_shard="mixed",
                    target_variant=target_variant,
                    device=device,
                )
                mixed_rows.append(row)
                optimizer_step_performed = optimizer_step_performed or performed

    within_summary = aggregate_eval_rows(within_rows, eval_type="within_shard")
    cross_summary = (
        aggregate_eval_rows(cross_rows, eval_type="cross_shard")
        if cross_rows
        else skipped_eval_summary("cross_shard", shard0_skip_reason or "shard0 gap0 artifacts unavailable")
    )
    mixed_summary = (
        aggregate_eval_rows(mixed_rows, eval_type="mixed_shard")
        if mixed_rows
        else skipped_eval_summary("mixed_shard", shard0_skip_reason or "shard0 gap0 artifacts unavailable")
    )
    step32_window_summary = _read_json(config["input"]["step32_run_dir"] + "/horizon_window_summary.json")
    step32_importance_summary = _read_json(config["input"]["step32_importance_summary_json"])
    shard0_gap0_ids = {str(record["sample_id"]) for record in read_jsonl(config["input"]["step32_horizon_window_manifest_jsonl"]) if int(record.get("horizon_gap", 0)) == 0}
    shard0_importance_distribution = summarize_step33b_proxy_importance(
        config["input"]["step32_importance_manifest_jsonl"],
        shard_name="shard0",
        topk=int(config["proxy_importance"].get("topk", 256)),
        allowed_sample_ids=shard0_gap0_ids,
    )
    shard1_importance_summary = _read_json(config["output"]["importance_summary_json"])
    shard0_summary = shard0_summary_from_step32(step32_window_summary, step32_importance_summary)
    shard0_summary["selected_windows"] = len(shard0_gap0_ids)
    shard1_window_summary = _read_json(config["output"]["shard1_window_summary_json"])
    dataset_bias = summarize_dataset_bias(
        shard0_summary=shard0_summary,
        shard1_summary=shard1_window_summary,
        shard0_importance_summary=shard0_importance_distribution,
        shard1_importance_summary=shard1_importance_summary,
    )
    decision_bits = decide_cross_shard_stability(
        shard1_summary=within_summary,
        cross_summary=cross_summary,
        mixed_summary=mixed_summary,
        min_proxy_beats_current_fraction=float(
            config["decision"]["stable_cross_shard_signal_requires"].get("proxy_beats_current_fraction_min", 0.67)
        ),
        min_proxy_beats_random_fraction=float(
            config["decision"]["stable_cross_shard_signal_requires"].get("proxy_beats_random_fraction_min", 0.67)
        ),
        require_cross_gain_positive=bool(
            config["decision"]["stable_cross_shard_signal_requires"].get("cross_shard_proxy_gain_positive", True)
        ),
    )
    decision = _decision(config, decision_bits, dataset_bias)
    trainval = {
        "stage": config["stage"],
        "safe_stop": False,
        "reason": None,
        "tiny_trainval_diagnosis_performed": True,
        "tiny_trainval_training_performed": True,
        "training_performed": True,
        "optimizer_step_performed": bool(optimizer_step_performed),
        "optimizer_step_scope": OPTIMIZER_SCOPE,
        "target_variant": target_variant,
        "split_seeds": [int(seed) for seed in config["evaluation"].get("split_seeds", [])],
        "shard1_num_samples": len(shard1_samples),
        "shard0_gap0_num_samples": len(shard0_samples),
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "videomae_training_performed": False,
        "selector_training_performed": False,
        "current_importance_training_performed": False,
        "action_conditioned_world_model_training_performed": False,
        "model_download_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "checkpoint_saved": False,
        "env_isaaclab_guard": env_guard,
        "safety_gate_pass": True,
    }
    payload = {
        "trainval_results": trainval,
        "within_shard_results": within_summary,
        "cross_shard_results": cross_summary,
        "mixed_shard_results": mixed_summary,
        "dataset_bias_summary": dataset_bias,
        "data_diversity_decision": decision,
    }
    _write_outputs(config, payload)
    return payload


def _run_policy_setting(
    *,
    config: dict[str, Any],
    train_cfg: dict[str, Any],
    policies: list[dict[str, Any]],
    train_samples: list[dict[str, Any]],
    val_samples: list[dict[str, Any]],
    split_seed: int,
    setting: str,
    train_shard: str,
    val_shard: str,
    target_variant: str,
    device: torch.device,
) -> tuple[dict[str, Any], bool]:
    policy_metrics: list[dict[str, Any]] = []
    policy_val: dict[str, float] = {}
    optimizer_step_performed = False
    for policy_index, policy in enumerate(policies):
        policy_cfg = dict(policy)
        if str(policy_cfg.get("selection")) == "random":
            policy_cfg["seed"] = int(policy_cfg.get("seed", 42)) + int(split_seed)
        _seed_everything(int(config.get("seed", 42)) + int(split_seed) + policy_index * 100)
        model = BridgeDataContextBottleneckSmokePredictor(
            hidden_dim=int(train_cfg.get("hidden_dim", 256)),
            seed=int(config.get("seed", 42)) + int(split_seed) + policy_index,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("learning_rate", 0.001)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        scope = _optimizer_scope(model, optimizer)
        if scope["optimizer_step_scope"] != OPTIMIZER_SCOPE:
            raise RuntimeError(f"optimizer scope must be {OPTIMIZER_SCOPE}: {scope}")
        train_data = _policy_dataset(train_samples, policy_cfg, target_variant, device)
        val_data = _policy_dataset(val_samples, policy_cfg, target_variant, device)
        curve, performed = _train_policy(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            val_data=val_data,
            train_steps=int(train_cfg.get("train_steps", 400)),
            eval_every=int(train_cfg.get("eval_every", 50)),
            grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
        )
        optimizer_step_performed = optimizer_step_performed or performed
        final_val = float(curve[-1]["val_loss"])
        policy_val[str(policy_cfg["name"])] = final_val
        policy_metrics.append(
            {
                "policy": str(policy_cfg["name"]),
                "val_final_loss": final_val,
                "val_best_loss": min(float(point["val_loss"]) for point in curve),
                "topk": train_data["topk"],
                "num_train_windows": len(train_samples),
                "num_val_windows": len(val_samples),
                "mean_selected_importance_mass_train": train_data["mean_selected_importance_mass"],
                "mean_selected_importance_mass_val": val_data["mean_selected_importance_mass"],
                "optimizer_step_performed": bool(performed),
                "optimizer_step_scope": OPTIMIZER_SCOPE,
            }
        )
    return (
        {
            "eval_type": setting,
            "setting": setting,
            "split_seed": int(split_seed),
            "train_shard": train_shard,
            "val_shard": val_shard,
            "target_variant": target_variant,
            "num_train_windows": len(train_samples),
            "num_val_windows": len(val_samples),
            "policy_metrics": policy_metrics,
            "policy_val": policy_val,
            "all_val_losses_finite": all(math.isfinite(float(value)) for value in policy_val.values()),
            "optimizer_step_scope": OPTIMIZER_SCOPE,
        },
        bool(optimizer_step_performed),
    )


def _load_shard1_samples(config: dict[str, Any]) -> list[dict[str, Any]]:
    samples = load_bridge_tfds_world_model_samples(_loader_config_for_outputs(config))
    for sample in samples:
        validate_world_model_sample(sample)
        sample["horizon_gap"] = 0
        sample["metadata"]["horizon_gap"] = 0
        sample["metadata"]["shard"] = "shard1"
    return samples


def _load_shard0_gap0_samples(config: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    try:
        gap0_ids = {
            str(record["sample_id"])
            for record in read_jsonl(config["input"]["step32_horizon_window_manifest_jsonl"])
            if int(record.get("horizon_gap", 0)) == 0
        }
        samples = load_bridge_tfds_world_model_samples(_loader_config_for_step32(config))
        filtered = []
        for sample in samples:
            if str(sample["sample_id"]) in gap0_ids:
                validate_world_model_sample(sample)
                sample["horizon_gap"] = 0
                sample["metadata"]["horizon_gap"] = 0
                sample["metadata"]["shard"] = "shard0"
                filtered.append(sample)
        if not filtered:
            return [], "Step32 gap0 token/importance samples were not found"
        return filtered, None
    except Exception as exc:
        return [], f"Step32 gap0 artifacts unavailable: {exc}"


def _load_shard0_gap0_splits(config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    splits = json.loads(Path(config["input"]["step32_horizon_splits_json"]).read_text(encoding="utf-8"))
    return {int(split["split_seed"]): split for split in splits.get("splits", []) if int(split.get("horizon_gap", -1)) == 0}


def _loader_config_for_outputs(config: dict[str, Any]) -> dict[str, Any]:
    loader = dict(config)
    loader["input"] = dict(config["input"])
    loader["input"].update(
        {
            "token_manifest_jsonl": config["output"]["token_manifest_jsonl"],
            "token_summary_json": config["output"]["token_summary_json"],
            "token_smoke_dir": config["output"]["token_smoke_dir"],
            "importance_manifest_jsonl": config["output"]["importance_manifest_jsonl"],
            "importance_summary_json": config["output"]["importance_summary_json"],
            "importance_smoke_dir": config["output"]["importance_dir"],
        }
    )
    loader["sample_limits"] = {"max_samples": _count_jsonl(config["output"]["token_manifest_jsonl"])}
    return loader


def _loader_config_for_step32(config: dict[str, Any]) -> dict[str, Any]:
    loader = dict(config)
    loader["input"] = dict(config["input"])
    loader["input"].update(
        {
            "token_manifest_jsonl": config["input"]["step32_frame_repeat_token_manifest_jsonl"],
            "token_summary_json": config["input"]["step32_frame_repeat_token_summary_json"],
            "token_smoke_dir": str(Path(config["input"]["step32_frame_repeat_token_manifest_jsonl"]).parent / "token_smoke"),
            "importance_manifest_jsonl": config["input"]["step32_importance_manifest_jsonl"],
            "importance_summary_json": config["input"]["step32_importance_summary_json"],
            "importance_smoke_dir": str(Path(config["input"]["step32_importance_manifest_jsonl"]).parent / "importance_smoke"),
        }
    )
    loader["sample_limits"] = {"max_samples": _count_jsonl(config["input"]["step32_frame_repeat_token_manifest_jsonl"])}
    return loader


def _samples_by_ids(by_id: dict[str, dict[str, Any]], sample_ids: list[str]) -> list[dict[str, Any]]:
    return [by_id[str(sample_id)] for sample_id in sample_ids if str(sample_id) in by_id]


def _decision(config: dict[str, Any], stability: dict[str, Any], dataset_bias: dict[str, Any]) -> dict[str, Any]:
    if stability["proxy_signal_stable_across_shards"]:
        recommended = {
            "name": "prepare proxy-supervised selector training plan after cross-shard validation",
            "condition": "proxy topK is stable on shard1, cross-shard, and mixed diagnostics",
            "scope": "no selector/current-importance training yet unless explicitly approved after review",
        }
    elif stability["shard1_proxy_signal_stable"] and not stability["cross_shard_proxy_signal_stable"]:
        recommended = {
            "name": "add more TFDS shards or shard-balanced training before selector",
            "condition": "proxy helps within shard1 but does not pass cross-shard stability",
            "scope": "no selector/current-importance training yet unless explicitly approved after review",
        }
    elif dataset_bias.get("dataset_bias_detected"):
        recommended = {
            "name": "diagnose dataset bias and representation before selector",
            "condition": "distribution shift or proxy-label shift is detected across shards",
            "scope": "no selector/current-importance training yet unless explicitly approved after review",
        }
    else:
        recommended = {
            "name": "improve representation or task target before selector",
            "condition": "proxy signal is not stable enough on the second shard",
            "scope": "no selector/current-importance training yet unless explicitly approved after review",
        }
    return {
        "stage": config["stage"],
        **stability,
        "dataset_bias_detected": bool(dataset_bias.get("dataset_bias_detected")),
        "distribution_shift_flags": dataset_bias.get("distribution_shift_flags", {}),
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "do_not_train_selector_yet": True,
        "do_not_train_current_importance_yet": True,
        "never_claim_final_utility": True,
        "recommended_step34": recommended,
        "safety_gate_pass": True,
    }


def _missing_inputs(config: dict[str, Any]) -> list[str]:
    required = [
        config["output"]["shard1_window_manifest_jsonl"],
        config["output"]["shard_split_json"],
        config["output"]["token_manifest_jsonl"],
        config["output"]["token_summary_json"],
        config["output"]["token_smoke_dir"],
        config["output"]["importance_manifest_jsonl"],
        config["output"]["importance_summary_json"],
        config["output"]["importance_dir"],
        config["input"]["step32_horizon_window_manifest_jsonl"],
        config["input"]["step32_horizon_splits_json"],
        config["input"]["step32_frame_repeat_token_manifest_jsonl"],
        config["input"]["step32_importance_manifest_jsonl"],
    ]
    return [str(path) for path in required if not Path(path).exists()]


def _safe_stop_payload(config: dict[str, Any], reason: str, env_guard: dict[str, bool]) -> dict[str, Any]:
    minimal = {
        "safe_stop": True,
        "reason": reason,
        "num_rows": 0,
        "rows": [],
        "safety_gate_pass": not (env_guard["tensorflow"] or env_guard["tensorflow_datasets"]),
    }
    decision = {
        "stage": config["stage"],
        "safe_stop": True,
        "reason": reason,
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "recommended_step34": {
            "name": "restore Step33B required inputs",
            "condition": reason,
            "scope": "no selector/current-importance training yet unless explicitly approved after review",
        },
        "safety_gate_pass": minimal["safety_gate_pass"],
    }
    return {
        "trainval_results": {
            "stage": config["stage"],
            "safe_stop": True,
            "reason": reason,
            "tiny_trainval_diagnosis_performed": False,
            "training_performed": False,
            "env_isaaclab_guard": env_guard,
            "safety_gate_pass": minimal["safety_gate_pass"],
        },
        "within_shard_results": {**minimal, "eval_type": "within_shard"},
        "cross_shard_results": {**minimal, "eval_type": "cross_shard"},
        "mixed_shard_results": {**minimal, "eval_type": "mixed_shard"},
        "dataset_bias_summary": {
            "stage": "bridgedata_v2_tfds_dataset_bias_step33b",
            "dataset_bias_detected": False,
            "distribution_shift_flags": {},
            "safety_gate_pass": minimal["safety_gate_pass"],
        },
        "data_diversity_decision": decision,
    }


def _write_outputs(config: dict[str, Any], payload: dict[str, Any]) -> None:
    _write_json(Path(config["output"]["within_shard_results_json"]), payload["within_shard_results"])
    _write_json(Path(config["output"]["cross_shard_results_json"]), payload["cross_shard_results"])
    _write_json(Path(config["output"]["mixed_shard_results_json"]), payload["mixed_shard_results"])
    _write_json(Path(config["output"]["dataset_bias_summary_json"]), payload["dataset_bias_summary"])
    _write_json(Path(config["output"]["data_diversity_decision_json"]), payload["data_diversity_decision"])


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _count_jsonl(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _env_guard() -> dict[str, bool]:
    return {
        "tensorflow": importlib.util.find_spec("tensorflow") is not None,
        "tensorflow_datasets": importlib.util.find_spec("tensorflow_datasets") is not None,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    payload = train_eval_step33b_data_diversity(args.config)
    print(json.dumps(payload["data_diversity_decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
