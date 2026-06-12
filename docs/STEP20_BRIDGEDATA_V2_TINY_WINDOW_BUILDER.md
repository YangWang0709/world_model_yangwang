# STEP20 BridgeData V2 Tiny-Subset Context Window Builder

## 1. Goal

Implement a BridgeData V2 tiny-subset window builder. This step does not train, download full data, extract tokens, or generate importance.

## 2. Why BridgeData V2

Step19 recommended BridgeData V2 as the first migration target and DROID as a future large-scale validation target.

## 3. What Was Not Done

- no full dataset download
- no model download
- no training
- no token extraction
- no importance generation
- no action-conditioned model
- no VLM/RL

## 4. Input Policy

- user-provided local tiny subset
- manifest.jsonl
- directory-per-trajectory inspection
- fake manifest dry-run when the local subset is missing

## 5. Manifest Format

See `docs/BRIDGEDATA_V2_USER_SUBSET_FORMAT.md`.

## 6. Window Spec

`context_len=16, current_len=4, future_len=4`

## 7. LongContextSample Output

Each output record is compatible with the Step19 `LongContextSample` schema and contains frame indices, optional frame path references, language/goal/action metadata references, and no video tensor.

## 8. Metadata Policy

Actions, language instructions, and goal images are saved as metadata only and are not model inputs.

## 9. Outputs

- window manifest JSONL
- subset inspection summary
- builder summary
- eval report

## 10. Tests

Step20 adds targeted config, manifest, builder, missing-subset, no-download, artifact-blacklist, and report tests. Full pytest should remain green.

## 11. Next Step

Step21 should do BridgeData V2 tiny-subset token extraction dry-run only if the user provides a real tiny subset or explicitly approves a tiny download.
