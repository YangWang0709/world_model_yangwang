# STEP22 BridgeData V2 User Subset Ingestion

Step22 adds a user-provided tiny-subset ingestion kit for BridgeData V2-style data.
Step21 safely stopped because no official <=1GB tiny sample with a known Content-Length
was available. This step does not download data; it prepares the project to accept a
small subset that the user manually places on disk.

## Goal

If a user places a tiny BridgeData V2-style subset under:

```text
data/bridgedata_v2_tiny_user_subset/
```

the project can:

- inspect the directory structure
- read or generate a normalized `manifest.jsonl`
- validate that each usable trajectory has at least 24 frames
- check that image paths exist without reading image payloads
- preserve action, language, camera, and goal-image fields as metadata only
- reuse the Step20 `build_bridgedata_windows_from_manifest()` window builder
- write a user-subset window manifest when the subset is valid

## Boundaries

- no BridgeData V2 download
- no DROID download
- no Open X-Embodiment download
- no model download
- no training
- no VideoMAE token extraction
- no importance generation
- no VLM, RL, Dreamer, TD-MPC, GDPO, or action-conditioned model
- no action, language, or goal image as model input

## Pending User Data

If the user subset directory is missing, Step22 reports:

```text
pending_user_data=true
real_format_validated=false
safety_gate_pass=true
```

This is an expected state, not a failure. The next step is for the user to prepare a
tiny local subset according to `docs/BRIDGEDATA_V2_USER_SUBSET_CHECKLIST.md`.

## Valid User Subset

If a valid user subset exists, Step22 reports:

```text
pending_user_data=false
real_format_validated=true
user_subset_window_manifest_exists=true
```

The recommended next step after validation is a BridgeData V2 user-subset token
extraction dry-run. That later step must be separately approved and should still stay
bounded to the user-provided tiny subset.

## Outputs

Runtime outputs are written under:

```text
runs/bridgedata_v2_user_subset_ingestion_step22_v1/
```

The committed source files include the config, scripts, tests, and docs only. The
`runs/` directory and real subset data remain ignored by git.
