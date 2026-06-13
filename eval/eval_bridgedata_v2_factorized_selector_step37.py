"""Evaluate and report Step37 factorized selector diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_factorized_selector_dataset_step37 import STAGE, load_step37_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_factorized_selector_step37.yaml"


def evaluate_step37_factorized_selector(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_step37_config(config_path)
    train_summary = _read_json(config["output"]["train_summary_json"])
    loss_curves = _read_json(config["output"]["loss_curves_json"])
    metrics = _read_json(config["output"]["metrics_json"])
    loss_ablation = _read_json(config["output"]["loss_ablation_json"])
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
        "factorized_selector_head_training_performed": bool(
            train_summary.get("factorized_selector_head_training_performed", False)
        ),
        "factorized_selector_with_proxy_temporal_prior_training_performed": bool(
            train_summary.get("factorized_selector_with_proxy_temporal_prior_training_performed", False)
        ),
        "loss_ablation_performed": bool(train_summary.get("loss_ablation_performed", False)),
        "optimizer_step_performed": bool(train_summary.get("optimizer_step_performed", False)),
        "optimizer_scope": train_summary.get("optimizer_scope"),
        "checkpoint_saved": False,
        "videomae_training_performed": False,
        "videomae_frozen": True,
        "current_importance_training_performed": False,
        "final_selector_training_performed": False,
        "world_model_training_performed": False,
        "downstream_task_training_performed": False,
        "within_package_eval_performed": int(within.get("num_rows", 0)) > 0,
        "cross_package_eval_performed": int(cross.get("num_rows", 0)) > 0,
        "mixed_package_eval_performed": int(mixed.get("num_rows", 0)) > 0,
        "within_package_metrics": _compact_metrics(within),
        "cross_package_metrics": _compact_metrics(cross),
        "mixed_package_metrics": _compact_metrics(mixed),
        "loss_ablation_summary": {
            "performed": bool(loss_ablation.get("performed", False)),
            "best_loss_variant": loss_ablation.get("best_loss_variant"),
            "best_factorized_selector_val_mse": loss_ablation.get("best_factorized_selector_val_mse"),
        },
        "factorized_selector_beats_random_token_baseline": bool(
            gate.get("factorized_selector_beats_random_token_baseline", False)
        ),
        "factorized_selector_beats_uniform_token_baseline": bool(
            gate.get("factorized_selector_beats_uniform_token_baseline", False)
        ),
        "factorized_selector_beats_temporal_broadcast_baseline": bool(
            gate.get("factorized_selector_beats_temporal_broadcast_baseline", False)
        ),
        "factorized_selector_beats_current_only_patch_baseline": bool(
            gate.get("factorized_selector_beats_current_only_patch_baseline", False)
        ),
        "factorized_selector_beats_step36_direct_patch_selector": bool(
            gate.get("factorized_selector_beats_step36_direct_patch_selector", False)
        ),
        "factorized_selector_generalizes_cross_shard": bool(
            gate.get("factorized_selector_generalizes_cross_shard", False)
        ),
        "top256_overlap_mean": float(gate.get("top256_overlap_mean", 0.0)),
        "future_factorized_selector_gate_ready": bool(gate.get("future_factorized_selector_gate_ready", False)),
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
        "recommended_step38": gate.get("recommended_step38", {}),
        "safety_gate_pass": bool(safety_gate),
    }
    _write_json(config["output"]["eval_json"], summary)
    Path(config["output"]["eval_md"]).write_text(_eval_markdown(summary), encoding="utf-8")
    _write_docs(train_summary, loss_curves, metrics, loss_ablation, gate, summary)
    return summary


def _safety_gate(train_summary: dict[str, Any], metrics: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        bool(train_summary.get("bounded_smoke_training_only", False))
        and bool(train_summary.get("factorized_selector_head_training_performed", False))
        and bool(train_summary.get("optimizer_step_performed", False))
        and train_summary.get("optimizer_scope") == "proxy_factorized_selector_head_only"
        and not bool(train_summary.get("videomae_training_performed", False))
        and bool(train_summary.get("videomae_frozen", True))
        and not bool(train_summary.get("current_importance_training_performed", False))
        and not bool(train_summary.get("final_selector_training_performed", False))
        and not bool(train_summary.get("world_model_training_performed", False))
        and not bool(train_summary.get("downstream_task_training_performed", False))
        and not bool(train_summary.get("checkpoint_saved", False))
        and not bool(train_summary.get("state_dict_saved", False))
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
        "factorized_selector_val_mse",
        "random_token_baseline_mse",
        "uniform_token_baseline_mse",
        "temporal_broadcast_baseline_mse",
        "current_only_patch_baseline_mse",
        "step36_direct_patch_selector_mse_if_available",
        "factorized_beats_random",
        "factorized_beats_uniform",
        "factorized_beats_temporal_broadcast",
        "factorized_beats_current_only",
        "factorized_beats_step36_direct_patch",
        "pearson",
        "spearman",
        "top64_overlap",
        "top128_overlap",
        "top256_overlap",
        "top512_overlap",
        "top256_precision",
        "top256_recall",
        "overfit_gap",
    ]
    return {key: payload.get(key) for key in keys}


def _write_docs(
    train_summary: dict[str, Any],
    loss_curves: dict[str, Any],
    metrics: dict[str, Any],
    loss_ablation: dict[str, Any],
    gate: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "STEP37_FACTORIZED_SELECTOR_DIAGNOSIS.md").write_text(
        "\n".join(
            [
                "# Step37 Factorized Selector Diagnosis",
                "",
                "Step37 exists because Step36 direct patch/token prediction beat random and uniform baselines, but did not beat temporal broadcast, current-only, or cross-package gates.",
                "",
                "Step37 trains only bounded factorized selector heads for diagnosis.",
                "Step37 does not train the final selector.",
                "Step37 does not train current importance.",
                "Step37 does not train VideoMAE.",
                "Step37 does not claim context utility.",
                "Step37 does not use downstream task improvement as evidence.",
                "",
                "Design:",
                "- temporal branch predicts frame scores `[16]`.",
                "- spatial residual branch predicts token residuals `[16,392]`.",
                "- combined score is temporal score plus spatial residual.",
                "- proxy temporal prior is used only as an auxiliary training target, not as model input.",
                "",
                "Safety:",
                f"- optimizer_scope: `{train_summary.get('optimizer_scope')}`",
                f"- checkpoint_saved: `{str(summary.get('checkpoint_saved')).lower()}`",
                f"- selector_training_allowed: `{str(summary.get('selector_training_allowed')).lower()}`",
                f"- current_importance_training_allowed: `{str(summary.get('current_importance_training_allowed')).lower()}`",
                f"- context_utility_claim_allowed: `{str(summary.get('context_utility_claim_allowed')).lower()}`",
                "- no writes to `data/" "token_shards/` or `data/" "importance_shards/`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    within = metrics.get("within_shard", {})
    cross = metrics.get("cross_shard", {})
    mixed = metrics.get("mixed_shard", {})
    (docs_dir / "BRIDGEDATA_V2_FACTORIZED_SELECTOR_REPORT.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Factorized Selector Report",
                "",
                f"- num_curves: `{loss_curves.get('num_curves')}`",
                f"- loss_ablation: `{json.dumps({'performed': loss_ablation.get('performed'), 'best_loss_variant': loss_ablation.get('best_loss_variant'), 'best_factorized_selector_val_mse': loss_ablation.get('best_factorized_selector_val_mse')}, sort_keys=True)}`",
                f"- within_package: `{json.dumps(_compact_metrics(within), sort_keys=True)}`",
                f"- cross_package: `{json.dumps(_compact_metrics(cross), sort_keys=True)}`",
                f"- mixed_package: `{json.dumps(_compact_metrics(mixed), sort_keys=True)}`",
                f"- factorized_selector_beats_temporal_broadcast_baseline: `{str(gate.get('factorized_selector_beats_temporal_broadcast_baseline')).lower()}`",
                f"- factorized_selector_beats_current_only_patch_baseline: `{str(gate.get('factorized_selector_beats_current_only_patch_baseline')).lower()}`",
                f"- factorized_selector_beats_step36_direct_patch_selector: `{str(gate.get('factorized_selector_beats_step36_direct_patch_selector')).lower()}`",
                f"- factorized_selector_generalizes_cross_shard: `{str(gate.get('factorized_selector_generalizes_cross_shard')).lower()}`",
                f"- top256_overlap_mean: `{gate.get('top256_overlap_mean')}`",
                "",
                "Action and language are metadata only. Future tokens are not selector inputs.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    recommended = gate.get("recommended_step38", {})
    (docs_dir / "BRIDGEDATA_V2_STEP37_SELECTOR_DECISION.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Step37 Selector Decision",
                "",
                f"- future_factorized_selector_gate_ready: `{str(gate.get('future_factorized_selector_gate_ready')).lower()}`",
                "- selector_training_allowed: `false`",
                "- final_selector_training_allowed: `false`",
                "- current_importance_training_allowed: `false`",
                "- context_utility_claim_allowed: `false`",
                "",
                "Recommended Step38:",
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
            "# Step37 Eval",
            "",
            f"- pass: `{str(summary.get('pass')).lower()}`",
            f"- safe_stop: `{str(summary.get('safe_stop')).lower()}`",
            f"- safety_gate_pass: `{str(summary.get('safety_gate_pass')).lower()}`",
            f"- optimizer_scope: `{summary.get('optimizer_scope')}`",
            f"- factorized_selector_beats_temporal_broadcast_baseline: `{str(summary.get('factorized_selector_beats_temporal_broadcast_baseline')).lower()}`",
            f"- factorized_selector_beats_current_only_patch_baseline: `{str(summary.get('factorized_selector_beats_current_only_patch_baseline')).lower()}`",
            f"- factorized_selector_beats_step36_direct_patch_selector: `{str(summary.get('factorized_selector_beats_step36_direct_patch_selector')).lower()}`",
            f"- factorized_selector_generalizes_cross_shard: `{str(summary.get('factorized_selector_generalizes_cross_shard')).lower()}`",
            f"- top256_overlap_mean: `{summary.get('top256_overlap_mean')}`",
            f"- future_factorized_selector_gate_ready: `{str(summary.get('future_factorized_selector_gate_ready')).lower()}`",
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
    print(json.dumps(evaluate_step37_factorized_selector(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
