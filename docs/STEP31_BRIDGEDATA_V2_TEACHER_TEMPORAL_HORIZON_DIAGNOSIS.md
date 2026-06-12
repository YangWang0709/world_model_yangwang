# Step31 BridgeData V2 Teacher Temporal Horizon Diagnosis

Step31 diagnoses why Step30B trained-predictor occlusion labels did not beat Step30A proxy labels.

- no selector training
- no current importance training
- no token extraction
- no teacher label regeneration
- no new data/model download

- pass: `true`
- recommended_step32: `{'name': 'longer-horizon BridgeData window builder and target diagnosis', 'condition': 'delta target exposes much stronger context gain under existing tokens', 'scope': 'no selector/current-importance training yet'}`
