# BridgeData V2 TFDS Context Sanity Report

- pass: `true`
- safe_stop: `false`
- context_sanity_performed: `true`
- step24_pass: `true`
- step25_pass: `true`
- step26_pass: `true`
- step27_pass: `true`
- num_samples: `4`
- current_only_can_overfit: `true`
- current_only_relative_loss_decrease: `0.9997079222742349`
- memorization_risk: `high`
- proxy_importance_mass_advantage_over_random: `6.600034242892526`
- context_utility_claim_allowed: `false`
- importance_label_is_proxy_only: `true`
- label_quality_note: `proxy dry-run only; not final teacher label`
- train_current_importance_now: `false`
- download_performed: `false`
- training_performed: `false`
- optimizer_step_performed: `false`
- token_extraction_performed: `false`
- importance_generation_performed: `false`
- data_token_shards_written: `false`
- data_importance_shards_written: `false`
- safety_gate_pass: `true`

## Interpretation

Step27 is best read as a tiny overfit sanity result. Because current_only also overfits almost perfectly, the experiment is memorization-prone and does not validate context utility. Step25 proxy importance shows ranking concentration, but it remains a proxy-only label.

## Context Utility Diagnosis

```json
{
  "allowed_claim": "proxy importance has ranking concentration, not validated predictive utility yet",
  "context_utility_claim_allowed": false,
  "current_only_can_overfit": true,
  "proxy_best_loss_close_to_full": true,
  "proxy_best_loss_relative_gap_to_full": 0.1939640960113298,
  "proxy_final_loss_better_than_random": false,
  "proxy_importance_mass_advantage_over_random": 6.600034242892526,
  "proxy_topk_selects_concentrated_context": true,
  "reason": "selected importance mass is concentrated, but tiny-overfit loss does not prove predictive advantage because sample count is 4 and model is random-init/tiny.",
  "sample_count": 4,
  "step26_selected_importance_mass": {
    "current_only": 0.0,
    "full_context_reference": 1.0,
    "proxy_importance_topk": 0.2452614637082408,
    "random_context_topk": 0.037160635033425635
  },
  "step27_loss_quality_note": "tiny-overfit only; not final performance",
  "step27_losses": {
    "current_only": {
      "best_loss": 0.0011004924308508635,
      "final_loss": 0.0035450139548629522,
      "initial_loss": 12.137228012084961,
      "relative_loss_decrease": 0.9997079222742349
    },
    "full_context_reference": {
      "best_loss": 0.0011100443080067635,
      "final_loss": 0.0011100443080067635,
      "initial_loss": 12.252837181091309,
      "relative_loss_decrease": 0.9999094051204958
    },
    "proxy_importance_topk": {
      "best_loss": 0.0013253530487418175,
      "final_loss": 0.07361903041601181,
      "initial_loss": 12.673670768737793,
      "relative_loss_decrease": 0.9941911832996634
    },
    "random_context_topk": {
      "best_loss": 0.0021274862810969353,
      "final_loss": 0.0021274862810969353,
      "initial_loss": 12.773147583007812,
      "relative_loss_decrease": 0.9998334407187209
    }
  }
}
```

## Memorization Risk

```json
{
  "generalization_not_tested": true,
  "memorization_risk": "high",
  "reason": "All policies, including current_only, can overfit almost perfectly on 4 samples.",
  "requires_train_val_split": true,
  "sample_count": 4
}
```

## Importance Label Diagnosis

```json
{
  "importance_method": "proxy_token_mse_dryrun",
  "importance_norm_max": 1.0,
  "importance_norm_min": 0.0,
  "importance_raw_mean": 5.381548930927238e-06,
  "importance_raw_std": 8.345384685526369e-06,
  "label_is_final_teacher": false,
  "label_is_proxy": true,
  "label_quality_note": "proxy dry-run only; not final teacher label",
  "normalized_signal_present": true,
  "raw_signal_small": true,
  "recommended_action": "use only for smoke; design stronger teacher before scaling selector training",
  "selected_mass_advantage_present": true
}
```
