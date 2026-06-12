"""Evaluate and report Step33B BridgeData TFDS data-diversity diagnostics."""

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

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_data_diversity_step33b.yaml"


def evaluate_step33b_data_diversity(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_yaml(config_path)
    outputs = config["output"]
    acquisition = _read_json(outputs["shard_acquisition_summary_json"])
    inventory = _read_json(outputs["shard_inventory_json"])
    window_summary = _read_json(outputs["shard1_window_summary_json"])
    token_summary = _read_json(outputs["token_summary_json"]) if Path(outputs["token_summary_json"]).exists() else {}
    importance_summary = _read_json(outputs["importance_summary_json"]) if Path(outputs["importance_summary_json"]).exists() else {}
    within = _read_json(outputs["within_shard_results_json"]) if Path(outputs["within_shard_results_json"]).exists() else {}
    cross = _read_json(outputs["cross_shard_results_json"]) if Path(outputs["cross_shard_results_json"]).exists() else {}
    mixed = _read_json(outputs["mixed_shard_results_json"]) if Path(outputs["mixed_shard_results_json"]).exists() else {}
    dataset_bias = _read_json(outputs["dataset_bias_summary_json"]) if Path(outputs["dataset_bias_summary_json"]).exists() else {}
    decision = _read_json(outputs["data_diversity_decision_json"]) if Path(outputs["data_diversity_decision_json"]).exists() else {}

    safe_stop = bool(acquisition.get("safe_stop") or window_summary.get("safe_stop") or token_summary.get("safe_stop"))
    second_shard_available = bool(acquisition.get("second_shard_available"))
    pass_ok = (
        (safe_stop and bool(acquisition.get("safety_gate_pass", True)))
        or (
            second_shard_available
            and int(window_summary.get("selected_windows", 0)) >= int(config["window_selection"].get("min_windows_per_shard", 32))
            and bool(token_summary.get("token_shapes_ok"))
            and bool(importance_summary.get("limited_proxy_importance_generation_performed"))
            and int(within.get("num_rows", 0)) > 0
        )
    )
    safety_gate = _safety_gate(acquisition, token_summary, importance_summary, decision)
    eval_summary = {
        "stage": config["stage"],
        "pass": bool(pass_ok and safety_gate and not safe_stop),
        "safe_stop": bool(safe_stop),
        "reason": acquisition.get("reason") or window_summary.get("reason") or token_summary.get("reason"),
        "second_shard_available": second_shard_available,
        "downloaded_new_shard_count": int(acquisition.get("downloaded_new_shard_count", 0)),
        "selected_shard_index": acquisition.get("selected_shard_index"),
        "selected_shard_filename": acquisition.get("selected_shard_filename"),
        "shard1_windows_selected": int(window_summary.get("selected_windows", 0)),
        "shard1_trajectories": int(window_summary.get("num_trajectories", 0)),
        "limited_token_extraction_performed": bool(token_summary.get("limited_token_extraction_performed")),
        "limited_proxy_importance_generation_performed": bool(
            importance_summary.get("limited_proxy_importance_generation_performed")
        ),
        "tiny_trainval_diagnosis_performed": int(within.get("num_rows", 0)) > 0,
        "shard1_proxy_beats_current_fraction": float(within.get("proxy_beats_current_fraction", 0.0) or 0.0),
        "shard1_proxy_beats_random_fraction": float(within.get("proxy_beats_random_fraction", 0.0) or 0.0),
        "shard1_proxy_gain_over_current_mean": float(within.get("proxy_gain_over_current_mean", 0.0) or 0.0),
        "cross_shard_eval_performed": bool(decision.get("cross_shard_eval_performed", False)),
        "cross_shard_proxy_signal_stable": bool(decision.get("cross_shard_proxy_signal_stable", False)),
        "mixed_shard_eval_performed": bool(decision.get("mixed_shard_eval_performed", False)),
        "dataset_bias_detected": bool(dataset_bias.get("dataset_bias_detected", False)),
        "distribution_shift_flags": dataset_bias.get("distribution_shift_flags", {}),
        "full_context_noise_confirmed": bool(
            within.get("full_context_noise_confirmed")
            or cross.get("full_context_noise_confirmed")
            or mixed.get("full_context_noise_confirmed")
        ),
        "context_utility_claim_allowed": False,
        "selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "raw_zip_downloaded": False,
        "full_tfds_downloaded": False,
        "model_download_performed": False,
        "videomae_training_performed": False,
        "selector_training_performed": False,
        "current_importance_training_performed": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "recommended_step34": decision.get("recommended_step34")
        or {
            "name": "restore Step33B required inputs",
            "condition": "Step33B safe-stopped before decision",
            "scope": "no selector/current-importance training yet unless explicitly approved after review",
        },
        "safety_gate_pass": bool(safety_gate),
    }
    _write_json(outputs["eval_json"], eval_summary)
    Path(outputs["eval_md"]).write_text(_eval_md(eval_summary), encoding="utf-8")
    _write_docs(config, acquisition, inventory, window_summary, token_summary, importance_summary, within, cross, mixed, dataset_bias, decision)
    return eval_summary


def _safety_gate(acquisition: dict[str, Any], token_summary: dict[str, Any], importance_summary: dict[str, Any], decision: dict[str, Any]) -> bool:
    return (
        int(acquisition.get("downloaded_new_shard_count", 0)) <= 1
        and not bool(acquisition.get("raw_zip_downloaded", False))
        and not bool(acquisition.get("full_tfds_downloaded", False))
        and not bool(token_summary.get("model_download_performed", False))
        and not bool(token_summary.get("training_performed", False))
        and not bool(importance_summary.get("teacher_training_performed", False))
        and not bool(importance_summary.get("selector_training_performed", False))
        and not bool(importance_summary.get("current_importance_generated", False))
        and not bool(decision.get("context_utility_claim_allowed", False))
        and not bool(decision.get("selector_training_allowed", False))
        and not bool(decision.get("current_importance_training_allowed", False))
    )


def _write_docs(
    config: dict[str, Any],
    acquisition: dict[str, Any],
    inventory: dict[str, Any],
    window_summary: dict[str, Any],
    token_summary: dict[str, Any],
    importance_summary: dict[str, Any],
    within: dict[str, Any],
    cross: dict[str, Any],
    mixed: dict[str, Any],
    dataset_bias: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    docs = PROJECT_ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "STEP33B_BRIDGEDATA_V2_TFDS_DATA_DIVERSITY.md").write_text(
        "\n".join(
            [
                "# Step33B BridgeData V2 TFDS Data Diversity",
                "",
                "Step33B diagnoses whether the Step32 frame-repeat proxy context gain is stable beyond a single TFDS shard.",
                "",
                f"- stage: `{config['stage']}`",
                f"- selected_shard_index: `{acquisition.get('selected_shard_index')}`",
                f"- downloaded_new_shard_count: `{acquisition.get('downloaded_new_shard_count', 0)}`",
                "- raw_zip_downloaded: `false`",
                "- full_tfds_downloaded: `false`",
                "- selector_training_allowed: `false`",
                "- current_importance_training_allowed: `false`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (docs / "BRIDGEDATA_V2_DATA_DIVERSITY_SHARD_REPORT.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Data Diversity Shard Report",
                "",
                f"- second_shard_available: `{str(acquisition.get('second_shard_available', False)).lower()}`",
                f"- selected_shard_filename: `{acquisition.get('selected_shard_filename')}`",
                f"- selected_windows: `{window_summary.get('selected_windows')}`",
                f"- shard1_trajectories: `{window_summary.get('num_trajectories')}`",
                f"- token_context_shape: `{token_summary.get('context_token_shape_example')}`",
                f"- importance_context_shape: `{importance_summary.get('context_importance_shape_example')}`",
                f"- local_shard_indices: `{inventory.get('local_shard_indices')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (docs / "BRIDGEDATA_V2_DATA_DIVERSITY_TARGET_DIAGNOSIS_REPORT.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Data Diversity Target Diagnosis",
                "",
                f"- shard1_proxy_beats_current_fraction: `{within.get('proxy_beats_current_fraction')}`",
                f"- shard1_proxy_beats_random_fraction: `{within.get('proxy_beats_random_fraction')}`",
                f"- shard1_proxy_gain_over_current_mean: `{within.get('proxy_gain_over_current_mean')}`",
                f"- cross_shard_eval_performed: `{str(decision.get('cross_shard_eval_performed', False)).lower()}`",
                f"- cross_shard_proxy_signal_stable: `{str(decision.get('cross_shard_proxy_signal_stable', False)).lower()}`",
                f"- mixed_shard_eval_performed: `{str(decision.get('mixed_shard_eval_performed', False)).lower()}`",
                f"- full_context_noise_within: `{str(within.get('full_context_noise_confirmed', False)).lower()}`",
                f"- full_context_noise_cross: `{str(cross.get('full_context_noise_confirmed', False)).lower()}`",
                f"- full_context_noise_mixed: `{str(mixed.get('full_context_noise_confirmed', False)).lower()}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (docs / "BRIDGEDATA_V2_DATASET_BIAS_DIAGNOSIS.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Dataset Bias Diagnosis",
                "",
                f"- dataset_bias_detected: `{str(dataset_bias.get('dataset_bias_detected', False)).lower()}`",
                f"- distribution_shift_flags: `{json.dumps(dataset_bias.get('distribution_shift_flags', {}), sort_keys=True)}`",
                f"- length_mean_relative_shift: `{dataset_bias.get('length_mean_relative_shift')}`",
                f"- proxy_topk_mass_relative_shift: `{dataset_bias.get('proxy_topk_mass_relative_shift')}`",
                f"- proxy_temporal_concentration_relative_shift: `{dataset_bias.get('proxy_temporal_concentration_relative_shift')}`",
                "- raw_language_text_saved: `false`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (docs / "BRIDGEDATA_V2_STEP33B_DECISION.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Step33B Decision",
                "",
                f"- proxy_signal_stable_across_shards: `{str(decision.get('proxy_signal_stable_across_shards', False)).lower()}`",
                f"- context_utility_claim_allowed: `{str(decision.get('context_utility_claim_allowed', False)).lower()}`",
                f"- selector_training_allowed: `{str(decision.get('selector_training_allowed', False)).lower()}`",
                f"- current_importance_training_allowed: `{str(decision.get('current_importance_training_allowed', False)).lower()}`",
                "",
                "Recommended Step34:",
                f"`{(decision.get('recommended_step34') or {}).get('name')}`",
                "",
                "Do not train selector yet. Do not train current importance yet. Do not claim final context utility.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _eval_md(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step33B Eval",
            "",
            f"- pass: `{str(summary.get('pass')).lower()}`",
            f"- safe_stop: `{str(summary.get('safe_stop')).lower()}`",
            f"- safety_gate_pass: `{str(summary.get('safety_gate_pass')).lower()}`",
            f"- selected_shard_index: `{summary.get('selected_shard_index')}`",
            f"- shard1_windows_selected: `{summary.get('shard1_windows_selected')}`",
            f"- cross_shard_proxy_signal_stable: `{str(summary.get('cross_shard_proxy_signal_stable')).lower()}`",
        ]
    ) + "\n"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    print(json.dumps(evaluate_step33b_data_diversity(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
