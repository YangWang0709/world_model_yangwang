# STEP19 Long-Context Dataset Feasibility

## 1. Why Step19

Step17 showed that the context bottleneck pipeline works. Step18 showed a negative BAIR loss-ablation result. Step17.5 audited the code path and passed. The likely issue is now dataset/task suitability: BAIR is too short and current-only context is already competitive.

## 2. What Was Not Done

- no training
- no full dataset download
- no model download
- no BAIR rerun
- no VLM/RL/action-conditioned model

## 3. Dataset Candidates

- BridgeData V2
- DROID
- Open X-Embodiment / RT-X collection as a listed-only future survey source

## 4. Feasibility Criteria

- trajectory length
- image observations
- action metadata
- language/goal metadata
- multi-camera metadata
- local subset feasibility
- storage requirement
- suitability for temporal/block context importance

## 5. Unified LongContextSample Schema

The schema records dataset name, split, trajectory/sample ids, context/current/future frame indices, optional video tensors, optional actions, end-effector state, language instruction, goal image, camera names, and metadata. Missing dataset-specific fields are allowed to be None.

## 6. Window Design

- short control: context_len=8, current_len=4, future_len=4
- medium: context_len=16, current_len=4, future_len=4
- long: context_len=32, current_len=4, future_len=8

## 7. BridgeData V2 Migration Plan

Use BridgeData V2 as the first migration target with a user-provided or metadata-only tiny subset of 100-500 trajectories. Keep actions, language, and goal images as metadata first.

## 8. DROID Migration Plan

Keep DROID as a future large-scale generalization target. It is richer and broader, but full use needs storage planning.

## 9. Recommendation

First target: BridgeData V2. Future large-scale target: DROID.

## 10. Risks

- official dataset format details still need tiny-subset verification
- BridgeData V2 full download is not appropriate for this local Step19 scope
- DROID full use likely requires external storage or cloud planning
- language/goal metadata could tempt a VLM path, but Step20 should keep it metadata-only

## 11. Next Step

Step20 should build a BridgeData V2 tiny-subset context window builder with no training, context_len=16, current_len=4, future_len=4, and temporal/block-level importance design.
