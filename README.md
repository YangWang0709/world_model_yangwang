# Task-Grounded Predictive Attention Bottleneck for Efficient World Models

This project studies task-grounded predictive token selection for efficient world models.

The goal is to select visual tokens that are task-relevant and useful for future latent prediction under limited token, latency, and memory budgets.

## Phase 1 Scope

Phase 1 is limited to:

1. Dataset loader.
2. Frozen video encoder token extraction.
3. Full-token teacher world model.
4. Attention student.
5. Baseline evaluation.
6. Visualization.

## Current Step 2 Scope

Step 2 only creates the engineering skeleton and a minimal smoke test.

This step does not include:

- Large model downloads.
- Training.
- Dataset downloads.
- V-JEPA, VideoMAE, or VLM weight downloads.
- Modifications to existing `/home/ubuntu22/VLA`, `/home/ubuntu22/MapExRL`, or `/home/ubuntu22/test_gpt` code.

