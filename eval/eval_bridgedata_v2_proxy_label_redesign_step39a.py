"""Evaluate and document Step39A proxy-label redesign diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bridgedata_v2_proxy_label_redesign_step39a import STAGE, load_step39a_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bridgedata_v2_tfds_proxy_label_redesign_step39a.yaml"


def evaluate_step39a_proxy_label_redesign(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_step39a_config(config_path)
    summary = _read_json(config["output"]["label_redesign_summary_json"])
    metrics = _read_json(config["output"]["label_variant_metrics_json"])
    comparison = _read_json(config["output"]["label_variant_comparison_json"])
    gate = _read_json(config["output"]["gate_decision_json"])
    safety_gate = _safety_gate(summary, metrics, comparison, gate)
    best = comparison.get("best_variant") or comparison.get("ranking", {}).get("best_variant") or {}
    result = {
        "stage": STAGE,
        "pass": bool(not summary.get("safe_stop", False) and summary.get("label_redesign_performed", False) and safety_gate),
        "safe_stop": bool(summary.get("safe_stop", False)),
        "reason": summary.get("reason"),
        "label_redesign_performed": bool(summary.get("label_redesign_performed", False)),
        "num_samples": int(summary.get("num_samples", 0)),
        "num_variants": int(summary.get("num_variants", 0)),
        "recommended_step40_label_variant": gate.get("recommended_step40_label_variant"),
        "recommended_step40_scope": gate.get("recommended_step40_scope"),
        "recommended_step40_reason": gate.get("recommended_step40_reason"),
        "best_variant": best,
        "redesigned_label_candidate_ready": bool(gate.get("redesigned_label_candidate_ready", False)),
        "selector_training_allowed": False,
        "final_selector_training_allowed": False,
        "current_importance_training_allowed": False,
        "context_utility_claim_allowed": False,
        "training_performed": False,
        "optimizer_step_performed": False,
        "checkpoint_saved": False,
        "state_dict_saved": False,
        "data_token_shards_written": False,
        "data_importance_shards_written": False,
        "new_dataset_download_performed": False,
        "new_model_download_performed": False,
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
    _write_docs(summary, metrics, comparison, gate, result)
    return result


def _safety_gate(
    summary: dict[str, Any],
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    gate: dict[str, Any],
) -> bool:
    false_flags = [
        summary.get("training_performed", False),
        summary.get("optimizer_step_performed", False),
        summary.get("checkpoint_saved", False),
        summary.get("state_dict_saved", False),
        summary.get("data_token_shards_written", False),
        summary.get("data_importance_shards_written", False),
        summary.get("new_dataset_download_performed", False),
        summary.get("new_model_download_performed", False),
        summary.get("action_used_as_input", False),
        summary.get("language_used_as_input", False),
        summary.get("future_tokens_used_as_input", False),
        metrics.get("training_performed", False),
        metrics.get("optimizer_step_performed", False),
        metrics.get("checkpoint_saved", False),
        metrics.get("state_dict_saved", False),
        comparison.get("redesigned_tensor_artifacts_saved", False),
        gate.get("selector_training_allowed", False),
        gate.get("final_selector_training_allowed", False),
        gate.get("current_importance_training_allowed", False),
        gate.get("context_utility_claim_allowed", False),
        gate.get("training_performed", False),
        gate.get("optimizer_step_performed", False),
        gate.get("checkpoint_saved", False),
        gate.get("state_dict_saved", False),
        gate.get("data_token_shards_written", False),
        gate.get("data_importance_shards_written", False),
        gate.get("new_dataset_download_performed", False),
        gate.get("new_model_download_performed", False),
        gate.get("action_used_as_input", False),
        gate.get("language_used_as_input", False),
        gate.get("future_tokens_used_as_input", False),
    ]
    return not any(bool(flag) for flag in false_flags)


def _write_docs(
    summary: dict[str, Any],
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    gate: dict[str, Any],
    result: dict[str, Any],
) -> None:
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    best = result.get("best_variant") or {}
    variants = metrics.get("variant_names", [])
    (docs_dir / "STEP39A_PROXY_LABEL_REDESIGN.md").write_text(
        "\n".join(
            [
                "# Step39A Proxy Label Redesign",
                "",
                "Step39A changes the proxy supervision target before training another selector.",
                "It is not selector training, because Step35, Step36, Step37, and Step38B showed that the old patch-level answer sheet was too noisy or not learnable enough against stronger baselines.",
                "",
                "Compared label variants:",
                ", ".join(f"`{name}`" for name in variants),
                "",
                "Variant meanings:",
                "- `temporal_only`: frame-level `[16]` target with a broadcast copy only for metric comparison.",
                "- `temporal_broadcast`: each frame score is copied to all 392 spatial tokens.",
                "- `coarse_index_bins_*` and `coarse_grid_*`: coarse token groups; grid variants are token-grid heuristics, not real pixel coordinates.",
                "- `denoised_soft_topk_*`: soft topK emphasis with a floor, not a hard one-hot mask.",
                "- global-prior variants test whether stable spatial bias explains the label.",
                "",
                f"Recommended Step40 label variant: `{gate.get('recommended_step40_label_variant')}`.",
                f"Reason: `{gate.get('recommended_step40_reason')}`.",
                "",
                "Safety:",
                "- trained selector/current importance/final selector/world model: `false`",
                "- downloaded new data/model: `false`",
                "- used action/language/future tokens as input: `false`",
                "- final context utility claim: `false`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    top_rows = comparison.get("ranking", {}).get("ranked_variants", [])[:6]
    (docs_dir / "BRIDGEDATA_V2_PROXY_LABEL_VARIANT_REPORT.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Proxy Label Variant Report",
                "",
                f"- num_samples: `{summary.get('num_samples')}`",
                f"- num_variants: `{summary.get('num_variants')}`",
                f"- global_spatial_prior_stats: `{json.dumps(summary.get('global_spatial_prior_stats', {}), sort_keys=True)}`",
                "",
                "Top diagnosis rows:",
                *[f"- `{json.dumps(row, sort_keys=True)}`" for row in top_rows],
                "",
                "The score is an engineering diagnosis heuristic, not a scientific proof that a selector is ready.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (docs_dir / "BRIDGEDATA_V2_STEP39A_DECISION.md").write_text(
        "\n".join(
            [
                "# BridgeData V2 Step39A Decision",
                "",
                f"- best_variant: `{best.get('variant_name')}`",
                f"- diagnosis_score: `{best.get('diagnosis_score')}`",
                f"- redesigned_label_candidate_ready: `{str(gate.get('redesigned_label_candidate_ready')).lower()}`",
                f"- recommended_step40_label_variant: `{gate.get('recommended_step40_label_variant')}`",
                f"- recommended_step40_scope: `{gate.get('recommended_step40_scope')}`",
                "- selector_training_allowed: `false`",
                "- final_selector_training_allowed: `false`",
                "- current_importance_training_allowed: `false`",
                "- context_utility_claim_allowed: `false`",
                "",
                "This step only redesigns proxy labels and compares their diagnostics. It does not prove final context utility.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _eval_markdown(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Step39A Eval",
            "",
            f"- pass: `{str(result.get('pass')).lower()}`",
            f"- safe_stop: `{str(result.get('safe_stop')).lower()}`",
            f"- safety_gate_pass: `{str(result.get('safety_gate_pass')).lower()}`",
            f"- recommended_step40_label_variant: `{result.get('recommended_step40_label_variant')}`",
            f"- recommended_step40_scope: `{result.get('recommended_step40_scope')}`",
            "- selector_training_allowed: `false`",
            "- final_selector_training_allowed: `false`",
            "- current_importance_training_allowed: `false`",
            "- context_utility_claim_allowed: `false`",
            "- optimizer_step_performed: `false`",
            "- checkpoint_saved: `false`",
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
    print(json.dumps(evaluate_step39a_proxy_label_redesign(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
