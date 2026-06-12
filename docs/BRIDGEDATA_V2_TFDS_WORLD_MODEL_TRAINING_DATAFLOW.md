# BridgeData V2 TFDS World-Model Training Dataflow

Step24 produces ignored run-dir token artifacts from BridgeData V2 TFDS `image_0` windows. Each sample contains frozen context tokens `[16, 392, 768]`, full current tokens `[4, 392, 768]`, and future tokens `[4, 392, 768]`.

Step25 reads those tokens and writes ignored proxy importance artifacts with context importance `[16, 392]`. The proxy labels are dry-run labels, not final teacher labels.

Step26 validates context bottleneck selection without training. The current tokens are always kept full. Context selection policies include `current_only`, `random_context_topk`, `proxy_importance_topk`, and `full_context_reference`.

Step27 reuses the same frozen Step24/25 samples and Step26 selection policies. For each policy, it initializes a separate tiny predictor, mean-pools the full current tokens and selected context tokens, and trains only that tiny predictor to match the future token summary. Loss curves are recorded per policy.

Forbidden modules remain untouched: VideoMAE, V-JEPA, ContextTeacher, importance teacher, selector, UnifiedPredictiveImportanceSelector, current importance, large world models, action-conditioned models, VLMs, and RL policies. Step27 writes only small JSON/MD summaries under the ignored run directory.
