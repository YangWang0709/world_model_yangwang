# BridgeData V2 Tiny Window Builder Report

- pass: `true`
- local subset exists: `false`
- manifest found: `false`
- used fake manifest: `true`
- num trajectories: `3`
- valid trajectories: `3`
- skipped trajectories: `0`
- generated windows: `51`
- window spec: `16/4/4`

## Boundaries

- no cloud required now
- no full BridgeData V2 download
- no DROID download
- no training
- no VideoMAE token extraction
- no importance generation
- no VLM/RL/action-conditioned model

## Metadata Policy

- action saved as metadata but not used as input
- language saved as metadata but not used as input
- goal image saved as metadata but not used as input

## Recommended Step21

- name: `BridgeData V2 tiny-subset token extraction dry-run`
- condition: `only if user provides a real tiny subset or approves tiny download`
- no_training: `true`
