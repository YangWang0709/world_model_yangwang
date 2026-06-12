# Step33A BridgeData V2 TFDS True-Temporal Token Diagnosis

Step33A follows Step32, where longer horizons did not increase context gain and the strongest frame-repeat setting remained gap0 with the future_delta_last_minus_current target.

This stage tests whether frame-repeat VideoMAE tokenization limited the representation by extracting true temporal clip tokens for the same Step32 gap0 windows.

- uses the existing BridgeData V2 TFDS mini shard
- reuses Step32 gap0 windows and splits
- uses the existing local VideoMAE model only
- trains only tiny diagnostic predictors
- does not train VideoMAE, selector, or current importance
- does not claim final context utility

- pass: `true`
- safe_stop: `false`
- true_temporal_representation_helped: `false`
