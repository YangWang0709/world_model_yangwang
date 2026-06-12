# STEP23 BridgeData V2 TFDS Mini-Shard Smoke

Step23 moves from raw BridgeData V2 downloads and user-provided subsets to a safer
official TFDS/RLDS mini-shard route.

Step21 safe-stopped because no official tiny raw sample with a known <=1GB size was
available. Step22 added a user-provided subset ingestion kit, but the user subset is
currently pending. Step23 therefore probes the official TFDS/RLDS directory, safely
downloads only a tiny shard subset, inspects real RLDS episode structure, and reuses
the existing Step20 long-context window builder.

## Why TFDS/RLDS Mini-Shard

- raw `demos_8_17.zip` and `scripted_6_18.zip` are forbidden
- full BridgeData V2 and full TFDS downloads are forbidden
- the official TFDS directory is sharded into roughly 90-130MB train records
- a single known-size shard can be used for a bounded local smoke
- the resulting episode metadata can be mapped into the project manifest format

## Environment Boundary

`env_isaaclab` must remain clean:

```text
tensorflow=false
tensorflow_datasets=false
```

All TensorFlow/TFDS work is run through:

```text
/home/ubuntu22/miniconda3/envs/tgpawb_tfds_py311/bin/python
```

If that environment is unavailable or lacks TensorFlow/TFDS, Step23 safe-stops and
does not install dependencies into `env_isaaclab`.

## What Step23 Does

- inventories the official TFDS directory listing
- selects at most a tiny known-size train shard subset under 1GB
- downloads only metadata and selected train shards
- reads real RLDS episode structure through the isolated TFDS environment
- converts valid episodes into BridgeData manifest records
- builds 16/4/4 context/current/future windows with the Step20 builder
- writes a report that clearly separates validated and safe-stop outcomes

## What Step23 Does Not Do

- no raw zip download
- no full TFDS download
- no DROID download
- no model download
- no training
- no VideoMAE token extraction
- no importance generation
- no VLM/RL/action-conditioned model
- no action/language/goal input path
- no image/video/tensor export

## Next Step

If `real_tfds_validated=true`, the recommended Step24 is a bounded BridgeData V2 TFDS
mini-shard token extraction dry-run. If Step23 safe-stops, fix the TFDS environment,
shard safety constraints, or move preparation to a cloud machine.
