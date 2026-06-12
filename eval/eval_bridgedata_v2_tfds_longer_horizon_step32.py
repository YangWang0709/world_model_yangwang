"""Evaluate Step32 BridgeData longer-horizon diagnostic outputs."""

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

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_longer_horizon_step32.yaml"
STEP_DOC = PROJECT_ROOT / "docs" / "STEP32_BRIDGEDATA_V2_TFDS_LONGER_HORIZON.md"
WINDOW_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_LONGER_HORIZON_WINDOW_REPORT.md"
TARGET_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_LONGER_HORIZON_TARGET_DIAGNOSIS_REPORT.md"
DECISION_DOC = PROJECT_ROOT / "docs" / "BRIDGEDATA_V2_STEP32_DECISION.md"


def evaluate_step32_longer_horizon(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    resolved = Path(config_path).resolve()
    config = _load_yaml(config_path)
    output = config["output"]
    windows = _read_json(output["horizon_window_summary_json"])
    splits = _read_json(output["horizon_split_json"])
    clip_summary = _read_json(output["clip_export_summary_json"])
    token_summary = _read_json(output["token_summary_json"])
    importance_summary = _read_json(output["importance_summary_json"])
    trainval = _read_json(output["trainval_results_json"])
    horizon_target = _read_json(output["horizon_target_summary_json"])
    decision = _read_json(output["step32_decision_json"])
    result = _result(config, windows, splits, clip_summary, token_summary, importance_summary, trainval, horizon_target, decision)

    _write_json(Path(output["eval_json"]), result)
    eval_report = _format_eval_report(result)
    Path(output["eval_md"]).parent.mkdir(parents=True, exist_ok=True)
    Path(output["eval_md"]).write_text(eval_report, encoding="utf-8")
    if resolved == DEFAULT_CONFIG.resolve():
        STEP_DOC.write_text(_format_step_doc(result), encoding="utf-8")
        WINDOW_DOC.write_text(_format_window_doc(result), encoding="utf-8")
        TARGET_DOC.write_text(_format_target_doc(result), encoding="utf-8")
        DECISION_DOC.write_text(_format_decision_doc(result), encoding="utf-8")
    return result


def _result(
    config: dict[str, Any],
    windows: dict[str, Any],
    splits: dict[str, Any],
    clip_summary: dict[str, Any],
    token_summary: dict[str, Any],
    importance_summary: dict[str, Any],
    trainval: dict[str, Any],
    horizon_target: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    horizons_completed = [int(gap) for gap in windows.get("horizons_completed", [])]
    required = [int(gap) for gap in config["window"]["required_horizon_gaps"]]
    optional = [int(gap) for gap in config["window"].get("optional_horizon_gaps", [])]
    forbidden = [
        windows.get("new_tfds_shard_downloaded"),
        windows.get("raw_zip_downloaded"),
        token_summary.get("new_tfds_shard_downloaded"),
        token_summary.get("model_download_performed"),
        token_summary.get("data_token_shards_written"),
        importance_summary.get("data_importance_shards_written"),
        trainval.get("new_tfds_shard_downloaded"),
        trainval.get("model_download_performed"),
        trainval.get("videomae_training_performed"),
        trainval.get("selector_training_performed"),
        trainval.get("current_importance_training_performed"),
        trainval.get("action_conditioned_world_model_training_performed"),
        trainval.get("data_token_shards_written"),
        trainval.get("data_importance_shards_written"),
        trainval.get("checkpoint_saved"),
        decision.get("context_utility_claim_allowed"),
        decision.get("selector_training_allowed"),
        decision.get("current_importance_training_allowed"),
    ]
    safety_gate_pass = not any(bool(item) for item in forbidden)
    safe_stop = bool(
        windows.get("safe_stop")
        or token_summary.get("safe_stop")
        or importance_summary.get("safe_stop")
        or trainval.get("safe_stop")
    )
    pass_ok = (
        safety_gate_pass
        and not safe_stop
        and all(gap in horizons_completed for gap in required)
        and any(gap > 0 for gap in horizons_completed)
        and bool(token_summary.get("token_shapes_ok"))
        and bool(importance_summary.get("limited_proxy_importance_generation_performed"))
        and bool(trainval.get("tiny_trainval_diagnosis_performed"))
        and bool(trainval.get("optimizer_step_performed"))
        and bool(decision.get("recommended_step33"))
        and not bool(decision.get("context_utility_claim_allowed"))
        and not bool(decision.get("selector_training_allowed"))
        and not bool(decision.get("current_importance_training_allowed"))
    )
    return {
        "stage": config["stage"],
        "pass": bool(pass_ok),
        "safe_stop": safe_stop,
        "longer_horizon_window_builder_performed": bool(windows.get("longer_horizon_window_builder_performed")),
        "limited_clip_export_performed": bool(clip_summary.get("clip_export_performed"))
        or bool(token_summary.get("limited_clip_export_performed")),
        "limited_token_extraction_performed": bool(token_summary.get("limited_token_extraction_performed")),
        "limited_proxy_importance_generation_performed": bool(
            importance_summary.get("limited_proxy_importance_generation_performed")
        ),
        "tiny_trainval_diagnosis_performed": bool(trainval.get("tiny_trainval_diagnosis_performed")),
        "horizons_completed": horizons_completed,
        "optional_horizons_completed": [gap for gap in optional if gap in horizons_completed],
        "selected_windows_by_horizon": {
            key: int(value.get("selected_windows", 0)) for key, value in windows.get("horizons", {}).items()
        },
        "trajectory_count_by_horizon": {
            key: int(value.get("num_trajectories", 0)) for key, value in windows.get("horizons", {}).items()
        },
        "split_seeds": splits.get("seeds", trainval.get("split_seeds", [])),
        "split_summaries": _split_summaries(splits),
        "token_shapes": {
            "context": token_summary.get("context_token_shape_example"),
            "current": token_summary.get("current_token_shape_example"),
            "future": token_summary.get("future_token_shape_example"),
        },
        "importance_shapes": {
            "context": importance_summary.get("context_importance_shape_example"),
            "temporal": importance_summary.get("temporal_importance_shape_example"),
            "spatial": importance_summary.get("spatial_importance_shape_example"),
        },
        "best_horizon_gap": horizon_target.get("best_horizon_gap"),
        "best_target_variant": horizon_target.get("best_target_variant"),
        "delta_target_amplifies_context_gain": bool(horizon_target.get("delta_target_amplifies_context_gain")),
        "delta_target_consistently_amplifies_context_gain": bool(
            horizon_target.get("delta_target_consistently_amplifies_context_gain")
        ),
        "proxy_beats_current_fraction": float(horizon_target.get("proxy_beats_current_fraction", 0.0) or 0.0),
        "proxy_beats_random_fraction": float(horizon_target.get("proxy_beats_random_fraction", 0.0) or 0.0),
        "full_context_noise_confirmed": bool(horizon_target.get("full_context_noise_confirmed")),
        "horizon_target_rows": horizon_target.get("horizon_target_rows", []),
        "per_horizon_target_val_table": _per_run_table(trainval),
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
        "recommended_step33": decision.get("recommended_step33") or horizon_target.get("recommended_step33"),
        "horizon_window_summary": windows,
        "token_summary": token_summary,
        "importance_summary": importance_summary,
        "trainval_results": trainval,
        "horizon_target_summary": horizon_target,
        "step32_decision": decision,
        "safety_gate_pass": bool(safety_gate_pass),
    }


def _split_summaries(splits: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "horizon_gap": int(split.get("horizon_gap", -1)),
            "split_seed": int(split.get("split_seed", -1)),
            "num_train_windows": int(split.get("num_train_windows", 0)),
            "num_val_windows": int(split.get("num_val_windows", 0)),
            "trajectory_disjoint": bool(split.get("train_val_trajectory_disjoint")),
            "fallback_used": bool(split.get("fallback_used")),
            "fallback_reason": split.get("fallback_reason"),
        }
        for split in splits.get("splits", [])
    ]


def _per_run_table(trainval: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in trainval.get("runs", []):
        row = {
            "horizon_gap": int(run.get("horizon_gap", -1)),
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
            "# BridgeData V2 Step32 Longer-Horizon Eval",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- safe_stop: `{str(result['safe_stop']).lower()}`",
            f"- horizons_completed: `{result['horizons_completed']}`",
            f"- best_horizon_gap: `{result['best_horizon_gap']}`",
            f"- best_target_variant: `{result['best_target_variant']}`",
            f"- delta_target_amplifies_context_gain: `{str(result['delta_target_amplifies_context_gain']).lower()}`",
            f"- proxy_beats_current_fraction: `{result['proxy_beats_current_fraction']}`",
            f"- proxy_beats_random_fraction: `{result['proxy_beats_random_fraction']}`",
            f"- full_context_noise_confirmed: `{str(result['full_context_noise_confirmed']).lower()}`",
            f"- recommended_step33: `{result['recommended_step33']}`",
            f"- safety_gate_pass: `{str(result['safety_gate_pass']).lower()}`",
            "",
        ]
    )


def _format_step_doc(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step32 BridgeData V2 TFDS Longer-Horizon Diagnosis",
            "",
            "Step32 follows Step31, where the delta target exposed a stronger context-gain signal than plain future summaries.",
            "",
            "- rebuilds longer-horizon windows from the existing BridgeData V2 TFDS mini shard",
            "- keeps context/current/future lengths fixed at 16/4/4",
            "- tests horizon gaps 0/4/8 and optional 12",
            "- performs limited clip export, frame-repeat VideoMAE token extraction, and proxy importance generation",
            "- trains only tiny diagnostic predictors",
            "- does not train selector or current importance",
            "- does not claim final context utility",
            "",
            f"- pass: `{str(result['pass']).lower()}`",
            f"- horizons_completed: `{result['horizons_completed']}`",
            f"- best_target_variant: `{result['best_target_variant']}`",
            "",
        ]
    )


def _format_window_doc(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 Longer-Horizon Window Report",
            "",
            f"- longer_horizon_window_builder_performed: `{str(result['longer_horizon_window_builder_performed']).lower()}`",
            f"- horizons_completed: `{result['horizons_completed']}`",
            f"- selected_windows_by_horizon: `{result['selected_windows_by_horizon']}`",
            f"- trajectory_count_by_horizon: `{result['trajectory_count_by_horizon']}`",
            f"- split_seeds: `{result['split_seeds']}`",
            f"- limited_clip_export_performed: `{str(result['limited_clip_export_performed']).lower()}`",
            f"- limited_token_extraction_performed: `{str(result['limited_token_extraction_performed']).lower()}`",
            f"- token_shapes: `{result['token_shapes']}`",
            f"- limited_proxy_importance_generation_performed: `{str(result['limited_proxy_importance_generation_performed']).lower()}`",
            f"- importance_shapes: `{result['importance_shapes']}`",
            f"- data_token_shards_written: `{str(result['data_token_shards_written']).lower()}`",
            f"- data_importance_shards_written: `{str(result['data_importance_shards_written']).lower()}`",
            "",
            "## Split Summary",
            "",
            "```json",
            json.dumps(result["split_summaries"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _format_target_doc(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# BridgeData V2 Longer-Horizon Target Diagnosis Report",
            "",
            f"- best_horizon_gap: `{result['best_horizon_gap']}`",
            f"- best_target_variant: `{result['best_target_variant']}`",
            f"- delta_target_amplifies_context_gain: `{str(result['delta_target_amplifies_context_gain']).lower()}`",
            f"- proxy_beats_current_fraction: `{result['proxy_beats_current_fraction']}`",
            f"- proxy_beats_random_fraction: `{result['proxy_beats_random_fraction']}`",
            f"- full_context_noise_confirmed: `{str(result['full_context_noise_confirmed']).lower()}`",
            f"- context_utility_claim_allowed: `{str(result['context_utility_claim_allowed']).lower()}`",
            f"- selector_training_allowed: `{str(result['selector_training_allowed']).lower()}`",
            f"- current_importance_training_allowed: `{str(result['current_importance_training_allowed']).lower()}`",
            "",
            "## Horizon Target Rows",
            "",
            "```json",
            json.dumps(result["horizon_target_rows"], indent=2, sort_keys=True),
            "```",
            "",
            "## Per Horizon/Target Validation Table",
            "",
            "```json",
            json.dumps(result["per_horizon_target_val_table"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def _format_decision_doc(result: dict[str, Any]) -> str:
    recommended = result.get("recommended_step33") or {}
    return "\n".join(
        [
            "# BridgeData V2 Step32 Decision",
            "",
            "Recommended Step33:",
            str(recommended.get("name")),
            "",
            "Do not claim final context utility.",
            "Do not train selector yet.",
            "Do not train current importance yet.",
            "",
            f"- condition: `{recommended.get('condition')}`",
            f"- scope: `{recommended.get('scope')}`",
            f"- best_horizon_gap: `{result['best_horizon_gap']}`",
            f"- best_target_variant: `{result['best_target_variant']}`",
            f"- delta_target_amplifies_context_gain: `{str(result['delta_target_amplifies_context_gain']).lower()}`",
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
    print(json.dumps(evaluate_step32_longer_horizon(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
