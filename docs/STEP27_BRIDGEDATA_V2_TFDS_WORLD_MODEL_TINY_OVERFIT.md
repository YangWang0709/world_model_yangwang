# Step27 BridgeData V2 TFDS World-Model Tiny Overfit

Step26 verified that real BridgeData V2 TFDS VideoMAE token artifacts can flow through the context bottleneck world-model forward/loss path. Step27 is the first stage that permits optimizer steps, but only for a tiny randomly initialized world-model predictor.

Allowed training is limited to `BridgeDataContextBottleneckSmokePredictor`. The Step24 VideoMAE tokens and Step25 proxy importance artifacts are read as frozen inputs. Current tokens stay fully preserved, context tokens are selected by policy, and the tiny predictor overfits four real token samples by predicting the future token summary.

This stage does not train VideoMAE, V-JEPA, ContextTeacher, importance teachers, selectors, current importance, action-conditioned models, VLMs, or RL policies. It does not download data or models, rerun token extraction, regenerate importance, or write token/importance shards.

The result is a tiny-overfit engineering check only. Loss decrease proves that the tiny predictor and optimizer plumbing are live on real token samples; it is not a final performance claim.
