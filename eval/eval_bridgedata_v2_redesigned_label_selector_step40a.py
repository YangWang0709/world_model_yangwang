"""Evaluate and report Step40A redesigned-label selector smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_redesigned_label_selector_dataset_step40a import OPTIMIZER_SCOPE, STAGE, load_step40a_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_redesigned_label_selector_step40a.yaml"


def evaluate_step40a_redesigned_label_selector(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_step40a_config(config_path)
    train_summary = _read_json(config["output"]["train_summary_json"])
    loss_curves = _read_json(config["output"]["loss_curves_json"])
    metrics = _read_json(config["output"]["selector_metrics_json"])
    gate = _read_json(config["output"]["gate_decision_json"])
    within = metrics.get("within_shard", {})
    cross = metrics.get("cross_shard", {})
    mixed = metrics.get("mixed_shard", {})
    safety_gate = _safety_gate(train_summary, metrics, gate)
    result = {
        "stage": STAGE,
        "pass": bool(not train_summary.get("safe_stop", False) and safety_gate and train_summary.get("training_performed", False)),
        "safe_stop": bool(train_summary.get("safe_stop", False)),
        "reason": train_summary.get("reason"),
        "label_variant": train_summary.get("label_variant"),
        "bounded_selector_smoke_training_performed": bool(
            train_summary.get("bounded_selector_smoke_training_performed", False)
        ),
        "optimizer_step_performed": bool(train_summary.get("optimizer_step_performed", False)),
        "optimizer_scope": train_summary.get("optimizer_scope"),
        "within_shard_metrics": _compact_metrics(within),
        "cross_shard_metrics": _compact_metrics(cross),
        "mixed_shard_metrics": _compact_metrics(mixed),
        "selector_beats_random_token_baseline": bool(gate.get("selector_beats_random_token_baseline", False)),
        "selector_beats_uniform_or_mean_baseline": bool(gate.get("selector_beats_uniform_or_mean_baseline", False)),
        "selector_beats_temporal_broadcast_baseline": bool(gate.get("selector_beats_temporal_broadcast_baseline", False)),
        "selector_beats_train_global_spatial_prior_baseline": bool(
            gate.get("selector_beats_train_global_spatial_prior_baseline", False)
        ),
        "selector_generalizes_cross_shard": bool(gate.get("selector_generalizes_cross_shard", False)),
        "top256_overlap_mean": float(gate.get("top256_overlap_mean", 0.0)),
        "overfit_gap_mean_abs": float(gate.get("overfit_gap_mean_abs", 0.0)),
        "redesigned_label_selector_smoke_pass": bool(gate.get("redesigned_label_selector_smoke_pass", False)),
        "recommended_step41": gate.get("recommended_step41", {}),
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
        "future_tokens_used_as_input": False,
        "tensorflow_required": False,
        "tensorflow_datasets_required": False,
        "extra_cloud_resources_used": False,
        "safety_gate_pass": bool(safety_gate),
    }
    _write_json(config["output"]["eval_json"], result)
    Path(config["output"]["eval_md"]).write_text(_eval_markdown(result), encoding="utf-8")
    _write_docs(train_summary, loss_curves, metrics, gate, result)
    return result


def _safety_gate(train_summary: dict[str, Any], metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        bool(train_summary.get("bounded_selector_smoke_training_performed", False))
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
        and not bool(train_summary.get("future_tokens_used_as_input", False))
        and not bool(gate.get("downstream_selector_use_allowed", False))
        and not bool(gate.get("final_selector_training_allowed", False))
        and not bool(gate.get("current_importance_training_allowed", False))
        and not bool(gate.get("context_utility_claim_allowed", False))
        and bool(metrics.get("safety_gate_pass", False))
    )


def _compact_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "num_rows",
        "selector_val_mse",
        "selector_val_mae",
        "random_token_baseline_mse",
        "uniform_or_mean_baseline_mse",
        "temporal_broadcast_baseline_mse",
        "train_global_spatial_prior_baseline_mse",
        "selector_beats_random_token_baseline",
        "selector_beats_uniform_or_mean_baseline",
        "selector_beats_temporal_broadcast_baseline",
        "selector_beats_train_global_spatial_prior_baseline",
        "pearson",
        "spearman",
        "top64_overlap",
        "top128_overlap",
        "top256_overlap",
        "top512_overlap",
        "top256_precision",
        "top256_recall",
        "train_mse",
        "overfit_gap",
    ]
    return {key: payload.get(key) for key in keys}


def _write_docs(
    train_summary: dict[str, Any],
    loss_curves: dict[str, Any],
    metrics: dict[str, Any],
    gate: dict[str, Any],
    result: dict[str, Any],
) -> None:
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "STEP40A_REDESIGNED_LABEL_SELECTOR_SMOKE.md").write_text(
        "\n".join(
            [
                "# Step40A Redesigned-Label Selector Smoke",
                "",
                "Step40A is a bounded smoke where a small student selector head learns from the redesigned answer sheet.",
                "It is not final selector training, because it does not permit downstream use and does not claim context utility.",
                "",
                f"- label_variant: `{result.get('label_variant')}`",
                "- train_global_spatial_prior is built from each train split only.",
                "- validation split samples are not used to construct that prior.",
                "- the same train prior is then applied to train and validation samples to build targets.",
                "",
                "Safety:",
                f"- bounded_selector_smoke_training_performed: `{str(result.get('bounded_selector_smoke_training_performed')).lower()}`",
                f"- optimizer_scope: `{result.get('optimizer_scope')}`",
                "- final selector/current importance/world model/downstream training: `false`",
                "- checkpoint/state_dict saved: `false`",
                "- new data/model downloads: `false`",
                "- action/language/future tokens used as selector input: `false`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    within = metrics.get("within_shard", {})
    cross = metrics.get("cross_shard", {})
    mixed = metrics.get("mixed_shard", {})
    (docs_dir / "BRIDGEDATA_V2_REDESIGNED_LABEL_SELECTOR_REPORT.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Redesigned-Label Selector Report",
                "",
                f"- num_curves: `{loss_curves.get('num_curves')}`",
                f"- within_shard: `{json.dumps(_compact_metrics(within), sort_keys=True)}`",
                f"- cross_shard: `{json.dumps(_compact_metrics(cross), sort_keys=True)}`",
                f"- mixed_shard: `{json.dumps(_compact_metrics(mixed), sort_keys=True)}`",
                f"- selector_beats_random_token_baseline: `{str(gate.get('selector_beats_random_token_baseline')).lower()}`",
                f"- selector_beats_uniform_or_mean_baseline: `{str(gate.get('selector_beats_uniform_or_mean_baseline')).lower()}`",
                f"- selector_beats_temporal_broadcast_baseline: `{str(gate.get('selector_beats_temporal_broadcast_baseline')).lower()}`",
                f"- selector_beats_train_global_spatial_prior_baseline: `{str(gate.get('selector_beats_train_global_spatial_prior_baseline')).lower()}`",
                f"- selector_generalizes_cross_shard: `{str(gate.get('selector_generalizes_cross_shard')).lower()}`",
                f"- top256_overlap_mean: `{gate.get('top256_overlap_mean')}`",
                f"- overfit_gap_mean_abs: `{gate.get('overfit_gap_mean_abs')}`",
                "",
                "All baselines are recomputed against the redesigned target label, not copied from Step36 or Step37.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    recommended = gate.get("recommended_step41", {})
    (docs_dir / "BRIDGEDATA_V2_STEP40A_DECISION.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Step40A Decision",
                "",
                f"- redesigned_label_selector_smoke_pass: `{str(gate.get('redesigned_label_selector_smoke_pass')).lower()}`",
                f"- downstream_selector_use_allowed: `{str(gate.get('downstream_selector_use_allowed')).lower()}`",
                "- final_selector_training_allowed: `false`",
                "- current_importance_training_allowed: `false`",
                "- context_utility_claim_allowed: `false`",
                "",
                "Recommended Step41:",
                f"- name: `{recommended.get('name')}`",
                f"- scope: `{recommended.get('scope')}`",
                "",
                "Step40A checks whether the redesigned label is easier for a lightweight selector to learn. It does not prove final context utility.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _eval_markdown(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step40A Eval",
            "",
            f"- pass: `{str(result.get('pass')).lower()}`",
            f"- safe_stop: `{str(result.get('safe_stop')).lower()}`",
            f"- safety_gate_pass: `{str(result.get('safety_gate_pass')).lower()}`",
            f"- label_variant: `{result.get('label_variant')}`",
            f"- optimizer_scope: `{result.get('optimizer_scope')}`",
            f"- redesigned_label_selector_smoke_pass: `{str(result.get('redesigned_label_selector_smoke_pass')).lower()}`",
            f"- recommended_step41: `{(result.get('recommended_step41') or {}).get('name')}`",
            "- downstream_selector_use_allowed: `false`",
            "- final_selector_training_allowed: `false`",
            "- current_importance_training_allowed: `false`",
            "- context_utility_claim_allowed: `false`",
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
    args = parser.parse_args()
    print(json.dumps(evaluate_step40a_redesigned_label_selector(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
