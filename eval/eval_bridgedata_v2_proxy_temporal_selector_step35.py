"""Evaluate and report Step35 proxy-temporal selector smoke training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_proxy_temporal_selector_dataset_step35 import STAGE, load_step35_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_proxy_temporal_selector_train_step35.yaml"


def evaluate_step35_proxy_temporal_selector(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_step35_config(config_path)
    train_summary = _read_json(config["output"]["train_summary_json"])
    loss_curves = _read_json(config["output"]["loss_curves_json"])
    metrics = _read_json(config["output"]["metrics_json"])
    gate = _read_json(config["output"]["gate_decision_json"])
    safe_stop = bool(train_summary.get("safe_stop", False))
    within = metrics.get("within_shard", {})
    cross = metrics.get("cross_shard", {})
    mixed = metrics.get("mixed_shard", {})
    safety_gate = _safety_gate(train_summary, metrics, gate)
    summary = {
        "stage": STAGE,
        "pass": bool(not safe_stop and safety_gate and bool(train_summary.get("training_performed", False))),
        "safe_stop": bool(safe_stop),
        "reason": train_summary.get("reason"),
        "training_performed": bool(train_summary.get("training_performed", False)),
        "bounded_smoke_training_only": True,
        "temporal_selector_head_training_performed": bool(
            train_summary.get("temporal_selector_head_training_performed", False)
        ),
        "optimizer_step_performed": bool(train_summary.get("optimizer_step_performed", False)),
        "optimizer_scope": train_summary.get("optimizer_scope"),
        "videomae_training_performed": False,
        "videomae_frozen": True,
        "current_importance_training_performed": False,
        "final_selector_training_performed": False,
        "patch_level_selector_training_performed": False,
        "checkpoint_saved": False,
        "within_shard_eval_performed": int(within.get("num_rows", 0)) > 0,
        "cross_shard_eval_performed": int(cross.get("num_rows", 0)) > 0,
        "mixed_shard_eval_performed": int(mixed.get("num_rows", 0)) > 0,
        "within_shard_metrics": _compact_metrics(within),
        "cross_shard_metrics": _compact_metrics(cross),
        "mixed_shard_metrics": _compact_metrics(mixed),
        "selector_beats_random_baseline": bool(gate.get("selector_beats_random_baseline", False)),
        "selector_beats_current_only_baseline": bool(gate.get("selector_beats_current_only_baseline", False)),
        "selector_generalizes_cross_shard": bool(gate.get("selector_generalizes_cross_shard", False)),
        "future_selector_training_gate_ready": bool(gate.get("future_selector_training_gate_ready", False)),
        "selector_training_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "context_utility_claim_allowed": False,
        "new_dataset_download_performed": False,
        "model_download_performed": False,
        "tensorflow_required": False,
        "tensorflow_datasets_required": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "recommended_step36": gate.get("recommended_step36", {}),
        "safety_gate_pass": bool(safety_gate),
    }
    _write_json(config["output"]["eval_json"], summary)
    Path(config["output"]["eval_md"]).write_text(_eval_markdown(summary), encoding="utf-8")
    _write_docs(config, train_summary, loss_curves, metrics, gate, summary)
    return summary


def _safety_gate(train_summary: dict[str, Any], metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        bool(train_summary.get("bounded_smoke_training_only", False))
        and bool(train_summary.get("temporal_selector_head_training_performed", False))
        and bool(train_summary.get("optimizer_step_performed", False))
        and train_summary.get("optimizer_scope") == "proxy_temporal_selector_head_only"
        and not bool(train_summary.get("videomae_training_performed", False))
        and bool(train_summary.get("videomae_frozen", True))
        and not bool(train_summary.get("current_importance_training_performed", False))
        and not bool(train_summary.get("final_selector_training_performed", False))
        and not bool(train_summary.get("patch_level_selector_training_performed", False))
        and not bool(train_summary.get("checkpoint_saved", False))
        and not bool(train_summary.get("data_token_shards_written", False))
        and not bool(train_summary.get("data_importance_shards_written", False))
        and not bool(gate.get("selector_training_allowed", False))
        and not bool(gate.get("final_selector_training_allowed", False))
        and not bool(gate.get("current_importance_training_allowed", False))
        and not bool(gate.get("context_utility_claim_allowed", False))
        and bool(metrics.get("safety_gate_pass", False))
    )


def _compact_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "num_rows",
        "selector_val_mse",
        "random_baseline_mse",
        "current_only_baseline_mse",
        "uniform_baseline_mse",
        "selector_beats_random_baseline",
        "selector_beats_current_only_baseline",
        "selector_beats_uniform_baseline",
        "selector_spearman",
        "selector_pearson",
        "top1_frame_hit",
        "top2_frame_overlap",
        "top4_frame_overlap",
        "overfit_gap",
    ]
    return {key: payload.get(key) for key in keys}


def _write_docs(
    config: dict[str, Any],
    train_summary: dict[str, Any],
    loss_curves: dict[str, Any],
    metrics: dict[str, Any],
    gate: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "STEP35_PROXY_TEMPORAL_SELECTOR_TRAINING_SMOKE.md").write_text(
        "\n".join(
            [
                "# Step35 Proxy Temporal Selector Training Smoke",
                "",
                "Step35 is a bounded temporal selector smoke, not a final experiment.",
                "",
                "Scope:",
                "- Step35 trains a temporal selector only.",
                "- Step35 does not train a patch-level selector.",
                "- Step35 does not train the final deployable selector.",
                "- Step35 does not train VideoMAE.",
                "- Step35 does not train current importance.",
                "- Step35 does not claim context utility.",
                "",
                "Safety flags:",
                f"- optimizer_scope: `{train_summary.get('optimizer_scope')}`",
                f"- checkpoint_saved: `{str(train_summary.get('checkpoint_saved')).lower()}`",
                f"- selector_training_allowed: `{str(summary.get('selector_training_allowed')).lower()}`",
                f"- current_importance_training_allowed: `{str(summary.get('current_importance_training_allowed')).lower()}`",
                f"- context_utility_claim_allowed: `{str(summary.get('context_utility_claim_allowed')).lower()}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    within = metrics.get("within_shard", {})
    cross = metrics.get("cross_shard", {})
    mixed = metrics.get("mixed_shard", {})
    (docs_dir / "BRIDGEDATA_V2_PROXY_TEMPORAL_SELECTOR_TRAINING_REPORT.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Proxy Temporal Selector Training Report",
                "",
                f"- num_curves: `{loss_curves.get('num_curves')}`",
                f"- within_shard: `{json.dumps(_compact_metrics(within), sort_keys=True)}`",
                f"- cross_shard: `{json.dumps(_compact_metrics(cross), sort_keys=True)}`",
                f"- mixed_shard: `{json.dumps(_compact_metrics(mixed), sort_keys=True)}`",
                f"- selector_beats_random_baseline: `{str(gate.get('selector_beats_random_baseline')).lower()}`",
                f"- selector_beats_current_only_baseline: `{str(gate.get('selector_beats_current_only_baseline')).lower()}`",
                f"- selector_generalizes_cross_shard: `{str(gate.get('selector_generalizes_cross_shard')).lower()}`",
                "",
                "The selector input is limited to existing context/current token summaries. Action and language are not model inputs.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    recommended = gate.get("recommended_step36", {})
    (docs_dir / "BRIDGEDATA_V2_STEP35_SELECTOR_DECISION.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Step35 Selector Decision",
                "",
                f"- future_selector_training_gate_ready: `{str(gate.get('future_selector_training_gate_ready')).lower()}`",
                "- selector_training_allowed: `false`",
                "- final_selector_training_allowed: `false`",
                "- current_importance_training_allowed: `false`",
                "- context_utility_claim_allowed: `false`",
                "",
                "Recommended Step36:",
                f"- name: `{recommended.get('name')}`",
                f"- scope: `{recommended.get('scope')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _eval_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step35 Eval",
            "",
            f"- pass: `{str(summary.get('pass')).lower()}`",
            f"- safe_stop: `{str(summary.get('safe_stop')).lower()}`",
            f"- safety_gate_pass: `{str(summary.get('safety_gate_pass')).lower()}`",
            f"- optimizer_scope: `{summary.get('optimizer_scope')}`",
            f"- future_selector_training_gate_ready: `{str(summary.get('future_selector_training_gate_ready')).lower()}`",
            "- selector_training_allowed: `false`",
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
    print(json.dumps(evaluate_step35_proxy_temporal_selector(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
