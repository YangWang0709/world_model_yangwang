# Long-Context Dataset Feasibility Report

Step19 is a feasibility and migration-planning step only. It performed no training, no model download, and no full dataset download.

## Current Project Conclusion

- Step17 validated the context bottleneck pipeline.
- Step18 showed that BAIR selector-loss changes did not improve downstream future MSE.
- Step17.5 audit passed, reducing the chance that the negative result is a simple implementation bug.
- Therefore, continuing BAIR loss ablations is lower value than moving to a longer-context dataset.

## Candidate Matrix

| Dataset | Role | Difficulty | Next action |
|---|---|---|---|
| BridgeData V2 | first local long-context migration target | medium | metadata-only dry-run, then user-provided 100-500 trajectory tiny subset |
| DROID | future large-scale generalization validation target | high | defer full use until storage plan; keep Step19 metadata-only |

## Recommendation

- first migration target: `BridgeData V2`
- future large-scale target: `DROID`
- Step20: `BridgeData V2 tiny-subset context window builder`

## Source Notes

- BridgeData V2 official page: https://rail-berkeley.github.io/bridgedata/
- BridgeData V2 paper page: https://proceedings.mlr.press/v229/walke23a.html
- BridgeData V2 official page/paper reports a large robot manipulation dataset with about 60k trajectories across 24 environments and goal/language-conditioned use cases.
- DROID official page: https://droid-dataset.github.io/
- DROID dataset docs: https://droid-dataset.github.io/droid/the-droid-dataset
- DROID paper page: https://arxiv.org/abs/2403.12945
- DROID official page/paper reports about 76k demonstration trajectories, 350h interaction data, 564 scenes, 86 tasks, and multi-camera data.
- Open X-Embodiment official page: https://robotics-transformer-x.github.io/
- Open X-Embodiment is listed only as a future survey source because it is too broad and large for the first local migration.

## Sanity Gates

- cloud_required_now: `false`
- full_download_performed: `false`
- large_download_performed: `false`
- training_performed: `false`
- sanity_gate_pass: `true`
