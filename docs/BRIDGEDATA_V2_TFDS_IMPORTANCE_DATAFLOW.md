# BridgeData V2 TFDS Importance Dataflow

Step25 starts from the Step24 token manifest:

```text
runs/bridgedata_v2_tfds_token_extraction_step24_v1/tfds_token_smoke_manifest.jsonl
```

Each manifest row points to a Step24 `.pt` token artifact containing:

- `context_tokens` with shape `[16, 392, 768]`
- `current_tokens` with shape `[4, 392, 768]`
- `future_tokens` with shape `[4, 392, 768]`

The Step25 loader validates those shapes, moves tensors to CPU `float32`, and
rejects any record that marks action, language, or goal as an input.

The proxy importance calculation summarizes future tokens, current tokens, and
context tokens in token space. It computes a base MSE prediction to the future
summary, then analytically estimates the loss delta caused by removing each
context token from the context mean. Positive deltas become the raw context
importance labels. A deterministic cosine-relevance fallback is used only if the
loss-delta signal is completely flat.

Step25 writes:

- small smoke `.pt` importance artifacts under the ignored `runs/` directory
- `tfds_importance_smoke_manifest.jsonl`
- `tfds_importance_summary.json`
- `tfds_importance_eval.json`
- `docs/BRIDGEDATA_V2_TFDS_IMPORTANCE_REPORT.md`

Teacher, selector, and world-model training are intentionally skipped. The goal
is only to validate the BridgeData token-to-importance plumbing before any
larger or training-dependent stage is considered.
