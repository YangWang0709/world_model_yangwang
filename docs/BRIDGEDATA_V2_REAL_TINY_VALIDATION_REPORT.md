# BridgeData V2 Real Tiny Validation Report

- safety_gate_pass: `true`
- real_format_validated: `false`
- safe_stop: `true`
- safe_stop_reason: `No official <=1GB tiny sample found or file size unknown.`
- download_performed: `false`
- download_bytes: `None`
- real_tiny_manifest_exists: `false`
- real_window_manifest_exists: `false`
- num_real_trajectories: `0`
- valid trajectories: `0`
- skipped trajectories: `0`
- frame count min/mean/max: `None/None/None`
- num_real_windows: `0`

## Metadata Fields

- images likely: `None`
- actions likely: `None`
- language likely: `None`
- goal image likely: `None`

## Boundaries

- no full BridgeData V2 download
- no DROID download
- no training
- no token extraction
- no importance generation
- action/language/goal image remain metadata only

## Recommended Step22

- name: `User-provided BridgeData tiny subset preparation`
- condition: `prepare a user-provided subset with manifest.jsonl and at least one trajectory with >=24 frames`
- no_training: `true`
