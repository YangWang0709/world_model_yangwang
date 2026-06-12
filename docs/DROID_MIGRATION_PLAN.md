# DROID Migration Plan

## Role

DROID is recommended as a future large-scale validation target, not the first local migration target.

## Why

- much larger and more diverse than BAIR or a BridgeData tiny subset
- useful for later generalization claims
- full local use likely needs storage planning, external disk, or cloud resources
- format choice between RLDS and raw data should be decided later

## Step19 Decision

- no full download
- no real sample download
- no training
- keep only schema mapping and feasibility notes

## Future Preconditions

- approved storage plan
- approved subset policy
- explicit decision on raw versus RLDS format
- separate no-leakage gate for language/action metadata if they become model inputs later
