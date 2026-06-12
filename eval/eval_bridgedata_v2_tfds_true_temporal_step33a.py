"""Evaluate Step33A true-temporal BridgeData representation outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_true_temporal_step33a.yaml"
STEP_DOC = PROJECT_ROOT / "docs" / "STEP33A_BRIDGEDATA_V2_TFDS_TRUE_TEMPORAL.md"
TOKEN_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TRUE_TEMPORAL_TOKEN_REPORT.md"
TARGET_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_TRUE_TEMPORAL_TARGET_DIAGNOSIS_REPORT.md"
DECISION_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_STEP33A_DECISION.md"


def evaluate_step33a_true_temporal(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    resolved = Path(config_path).resolve()
    config = _load_yaml(config_path)
    output = config["output"]
    token_summary = _read_json(output["true_temporal_token_summary_json"])
    importance_summary = _read_json(output["true_temporal_importance_summary_json"])
    trainval = _read_json(output["true_temporal_trainval_results_json"])
    comparison = _read_json(output["representation_comparison_json"])
    decision = _read_json(output["step33a_decision_json"])
    result = _result(config, token_summary, importance_summary, trainval, comparison, decision)
    _write_json(Path(output["eval_json"]), result)
    Path(output["eval_md"]).parent.mkdir(parents=True, exist_ok=True)
    Path(output["eval_md"]).write_text(_format_eval_report(result), encoding="utf-8")
    if resolved == DEFAULT_CONFIG.resolve():
        STEP_DOC.write_text(_format_step_doc(result), encoding="utf-8")
        TOKEN_DOC.write_text(_format_token_doc(result), encoding="utf-8")
        TARGET_DOC.write_text(_format_target_doc(result), encoding="utf-8")
        DECISION_DOC.write_text(_format_decision_doc(result), encoding="utf-8")
    return result


def _result(
    config: dict[str, Any],
    token_summary: dict[str, Any],
    importance_summary: dict[str, Any],
    trainval: dict[str, Any],
    comparison: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    forbidden = [
        token_summary.get("model_download_performed"),
        token_summary.get("videomae_training_performed"),
        token_summary.get("data_token_shards_written"),
        token_summary.get("large_token_shards_generated"),
        importance_summary.get("model_download_performed"),
        importance_summary.get("training_performed"),
        importance_summary.get("selector_training_performed"),
        importance_summary.get("train_current_importance"),
        importance_summary.get("current_importance_generated"),
        importance_summary.get("data_importance_shards_written"),
        trainval.get("new_tfds_shard_downloaded"),
        trainval.get("raw_zip_downloaded"),
        trainval.get("full_tfds_downloaded"),
        trainval.get("droid_downloaded"),
        trainval.get("model_download_performed"),
        trainval.get("videomae_training_performed"),
        trainval.get("selector_training_performed"),
        trainval.get("current_importance_training_performed"),
        trainval.get("action_conditioned_world_model_training_performed"),
        trainval.get("data_token_shards_written"),
        trainval.get("data_importance_shards_written"),
        trainval.get("checkpoint_saved"),
        comparison.get("context_utility_claim_allowed"),
        comparison.get("selector_training_allowed"),
        comparison.get("current_importance_training_allowed"),
        decision.get("context_utility_claim_allowed"),
        decision.get("selector_training_allowed"),
        decision.get("current_importance_training_allowed"),
    ]
    safety_gate_pass = not any(bool(item) for item in forbidden)
    safe_stop = bool(token_summary.get("safe_stop") or importance_summary.get("safe_stop") or trainval.get("safe_stop"))
    pass_ok = (
        safety_gate_pass
        and not safe_stop
        and bool(token_summary.get("limited_true_temporal_token_extraction_performed"))
        and bool(importance_summary.get("limited_true_temporal_proxy_importance_performed"))
        and bool(trainval.get("tiny_trainval_diagnosis_performed"))
        and bool(trainval.get("optimizer_step_performed"))
        and bool(decision.get("recommended_step33b_or_step34"))
        and not bool(decision.get("context_utility_claim_allowed"))
        and not bool(decision.get("selector_training_allowed"))
        and not bool(decision.get("current_importance_training_allowed"))
    )
    return {
        "stage": config["stage"],
        "pass": bool(pass_ok),
        "safe_stop": bool(safe_stop),
        "limited_true_temporal_token_extraction_performed": bool(
            token_summary.get("limited_true_temporal_token_extraction_performed")
        ),
        "limited_true_temporal_proxy_importance_performed": bool(
            importance_summary.get("limited_true_temporal_proxy_importance_performed")
        ),
        "tiny_trainval_diagnosis_performed": bool(trainval.get("tiny_trainval_diagnosis_performed")),
        "horizon_gap": 0,
        "target_variant": config["comparison"]["primary_target"],
        "num_samples": int(token_summary.get("num_samples", 0) or 0),
        "raw_token_shapes": token_summary.get("raw_token_shapes"),
        "summary_shapes": token_summary.get("summary_shapes"),
        "token_shapes": {
            "context": token_summary.get("context_token_shape_example"),
            "current": token_summary.get("current_token_shape_example"),
            "future": token_summary.get("future_token_shape_example"),
        },
        "importance_shape": importance_summary.get("context_importance_shape_example"),
        "importance_shapes": {
            "context": importance_summary.get("context_importance_shape_example"),
            "temporal": importance_summary.get("temporal_importance_shape_example"),
            "spatial": importance_summary.get("spatial_importance_shape_example"),
        },
        "temporal_spatial_summary_available": bool(importance_summary.get("temporal_spatial_summary_available")),
        "frame_repeat_proxy_gain_mean": float(comparison.get("frame_repeat_proxy_gain_mean", 0.0) or 0.0),
        "true_temporal_proxy_gain_mean": float(comparison.get("true_temporal_proxy_gain_mean", 0.0) or 0.0),
        "true_temporal_gain_over_frame_repeat": float(
            comparison.get("true_temporal_gain_over_frame_repeat", 0.0) or 0.0
        ),
        "true_temporal_representation_helped": bool(comparison.get("true_temporal_representation_helped")),
        "true_temporal_proxy_beats_current_fraction": float(
            comparison.get("true_temporal_proxy_beats_current_fraction", 0.0) or 0.0
        ),
        "true_temporal_proxy_beats_random_fraction": float(
            comparison.get("true_temporal_proxy_beats_random_fraction", 0.0) or 0.0
        ),
        "full_context_noise_confirmed": bool(comparison.get("full_context_noise_confirmed")),
        "target_rows": comparison.get("target_rows", []),
        "per_seed_val_table": _per_seed_val_table(trainval),
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "new_tfds_shard_downloaded": False,
        "raw_zip_downloaded": False,
        "full_tfds_downloaded": False,
        "droid_downloaded": False,
        "model_download_performed": False,
        "videomae_training_performed": False,
        "selector_training_performed": False,
        "current_importance_training_performed": False,
        "action_conditioned_world_model_training_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "checkpoint_saved": False,
        "recommended_step33b_or_step34": decision.get("recommended_step33b_or_step34")
        or comparison.get("recommended_step33b_or_step34"),
        "token_summary": token_summary,
        "importance_summary": importance_summary,
        "true_temporal_trainval_results": trainval,
        "representation_comparison": comparison,
        "step33a_decision": decision,
        "safety_gate_pass": bool(safety_gate_pass),
    }


def _per_seed_val_table(trainval: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in trainval.get("runs", []):
        row = {
            "representation_mode": str(run.get("representation_mode")),
            "horizon_gap": int(run.get("horizon_gap", 0)),
            "target_variant": str(run.get("target_variant")),
            "split_seed": int(run.get("split_seed", -1)),
            "num_train_windows": int(run.get("num_train_windows", 0)),
            "num_val_windows": int(run.get("num_val_windows", 0)),
            "trajectory_disjoint": bool(run.get("trajectory_disjoint")),
            **{key: float(value) for key, value in (run.get("policy_val") or {}).items()},
            "proxy_gain_over_current": float(run.get("proxy_gain_over_current", 0.0)),
            "proxy_beats_current": bool(run.get("proxy_beats_current")),
            "proxy_beats_random": bool(run.get("proxy_beats_random")),
            "full_beats_current": bool(run.get("full_beats_current")),
        }
        rows.append(row)
    return rows


def _format_eval_report(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 Step33A True-Temporal Eval",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- safe_stop: `{str(result['safe_stop']).lower()}`",
            f"- num_samples: `{result['num_samples']}`",
            f"- frame_repeat_proxy_gain_mean: `{result['frame_repeat_proxy_gain_mean']}`",
            f"- true_temporal_proxy_gain_mean: `{result['true_temporal_proxy_gain_mean']}`",
            f"- true_temporal_gain_over_frame_repeat: `{result['true_temporal_gain_over_frame_repeat']}`",
            f"- true_temporal_representation_helped: `{str(result['true_temporal_representation_helped']).lower()}`",
            f"- true_temporal_proxy_beats_current_fraction: `{result['true_temporal_proxy_beats_current_fraction']}`",
            f"- true_temporal_proxy_beats_random_fraction: `{result['true_temporal_proxy_beats_random_fraction']}`",
            f"- full_context_noise_confirmed: `{str(result['full_context_noise_confirmed']).lower()}`",
            f"- recommended_step33b_or_step34: `{result['recommended_step33b_or_step34']}`",
            f"- safety_gate_pass: `{str(result['safety_gate_pass']).lower()}`",
            "",
        ]
    )


def _format_step_doc(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step33A BridgeData V2 TFDS True-Temporal Token Diagnosis",
            "",
            "Step33A follows Step32, where longer horizons did not increase context gain and the strongest frame-repeat setting remained gap0 with the future_delta_last_minus_current target.",
            "",
            "This stage tests whether frame-repeat VideoMAE tokenization limited the representation by extracting true temporal clip tokens for the same Step32 gap0 windows.",
            "",
            "- uses the existing BridgeData V2 TFDS mini shard",
            "- reuses Step32 gap0 windows and splits",
            "- uses the existing local VideoMAE model only",
            "- trains only tiny diagnostic predictors",
            "- does not train VideoMAE, selector, or current importance",
            "- does not claim final context utility",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- safe_stop: `{str(result['safe_stop']).lower()}`",
            f"- true_temporal_representation_helped: `{str(result['true_temporal_representation_helped']).lower()}`",
            "",
        ]
    )


def _format_token_doc(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 True-Temporal Token Report",
            "",
            f"- tokenization_mode: `true_temporal_clip`",
            f"- limited_true_temporal_token_extraction_performed: `{str(result['limited_true_temporal_token_extraction_performed']).lower()}`",
            f"- num_samples: `{result['num_samples']}`",
            f"- raw_token_shapes: `{result['raw_token_shapes']}`",
            f"- token_shapes: `{result['token_shapes']}`",
            f"- summary_shapes: `{result['summary_shapes']}`",
            f"- temporal_spatial_summary_available: `{str(result['temporal_spatial_summary_available']).lower()}`",
            f"- model_download_performed: `{str(result['model_download_performed']).lower()}`",
            f"- videomae_training_performed: `{str(result['videomae_training_performed']).lower()}`",
            f"- data_token_shards_written: `{str(result['data_token_shards_written']).lower()}`",
            "",
            "## Token Summary JSON",
            "",
            "```json",
            json.dumps(result["token_summary"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _format_target_doc(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 True-Temporal Target Diagnosis Report",
            "",
            f"- target_variant: `{result['target_variant']}`",
            f"- frame_repeat_proxy_gain_mean: `{result['frame_repeat_proxy_gain_mean']}`",
            f"- true_temporal_proxy_gain_mean: `{result['true_temporal_proxy_gain_mean']}`",
            f"- true_temporal_gain_over_frame_repeat: `{result['true_temporal_gain_over_frame_repeat']}`",
            f"- true_temporal_proxy_beats_current_fraction: `{result['true_temporal_proxy_beats_current_fraction']}`",
            f"- true_temporal_proxy_beats_random_fraction: `{result['true_temporal_proxy_beats_random_fraction']}`",
            f"- full_context_noise_confirmed: `{str(result['full_context_noise_confirmed']).lower()}`",
            f"- true_temporal_representation_helped: `{str(result['true_temporal_representation_helped']).lower()}`",
            f"- context_utility_claim_allowed: `{str(result['context_utility_claim_allowed']).lower()}`",
            f"- selector_training_allowed: `{str(result['selector_training_allowed']).lower()}`",
            f"- current_importance_training_allowed: `{str(result['current_importance_training_allowed']).lower()}`",
            "",
            "## Target Rows",
            "",
            "```json",
            json.dumps(result["target_rows"], indent=2, sort_keys=True),
            "```",
            "",
            "## Per-Seed Validation Table",
            "",
            "```json",
            json.dumps(result["per_seed_val_table"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _format_decision_doc(result: dict[str, Any]) -> str:
    recommended = result.get("recommended_step33b_or_step34") or {}
    return "\n".join(
        [
            "# BridgeData V2 Step33A Decision",
            "",
            "Recommended Step33B/Step34:",
            str(recommended.get("name")),
            "",
            "Do not claim final context utility.",
            "Do not train selector yet.",
            "Do not train current importance yet.",
            "",
            f"- condition: `{recommended.get('condition')}`",
            f"- scope: `{recommended.get('scope')}`",
            f"- true_temporal_representation_helped: `{str(result['true_temporal_representation_helped']).lower()}`",
            f"- frame_repeat_proxy_gain_mean: `{result['frame_repeat_proxy_gain_mean']}`",
            f"- true_temporal_proxy_gain_mean: `{result['true_temporal_proxy_gain_mean']}`",
            f"- true_temporal_gain_over_frame_repeat: `{result['true_temporal_gain_over_frame_repeat']}`",
            f"- context_utility_claim_allowed: `{str(result['context_utility_claim_allowed']).lower()}`",
            f"- selector_training_allowed: `{str(result['selector_training_allowed']).lower()}`",
            f"- current_importance_training_allowed: `{str(result['current_importance_training_allowed']).lower()}`",
            "",
        ]
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    print(json.dumps(evaluate_step33a_true_temporal(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
