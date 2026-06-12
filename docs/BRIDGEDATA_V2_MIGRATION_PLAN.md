# BridgeData V2 Migration Plan

## Role

BridgeData V2 is recommended as the first long-context migration target.

## Why

- broader and longer than BAIR for manipulation behavior
- closer to local tiny-subset feasibility than DROID
- goal image and language metadata can support later task-grounded conditioning
- Step20 still does not connect VLM or a language encoder

## Initial Subset

- trajectories: `100-500 trajectories`
- context_len: `16`
- current_len: `4`
- future_len: `4`
- encoder: `existing local VideoMAE first`
- action/language/goal: metadata only
- training: `false`

## Required Answers Before Full Migration

- exact tiny-subset source path
- per-trajectory frame count distribution
- consistency of goal-image and language fields
- final local disk budget for preprocessed windows and tokens
