# BridgeData V2 TFDS Context Bottleneck Dataflow

Step26 reads two ignored run directories:

- `runs/bridgedata_v2_tfds_token_extraction_step24_v1/`
- `runs/bridgedata_v2_tfds_importance_step25_v1/`

The batch builder aligns Step24 token artifacts and Step25 importance artifacts
by `sample_id`. Each sample contains:

- context tokens `[16, 392, 768]`
- current tokens `[4, 392, 768]`
- future tokens `[4, 392, 768]`
- context importance `[16, 392]`

The context selection module flattens context tokens to `[6272, 768]` and
context importance to `[6272]`. `current_only` selects zero context tokens,
`random_context_topk` selects a deterministic random topK, `proxy_importance_topk`
selects the highest Step25 proxy-importance tokens, and
`full_context_reference` keeps all context tokens as a non-deployable reference.

The smoke predictor mean-pools full current tokens and selected context tokens,
concatenates both summaries, and predicts the future-token summary. The MSE loss
is checked only for finiteness.

Action, language, and goal fields remain metadata only.
