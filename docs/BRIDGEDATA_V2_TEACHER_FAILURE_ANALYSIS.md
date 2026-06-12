# BridgeData V2 Teacher Failure Analysis

- teacher_topk_underperforms_proxy_confirmed: `true`
- teacher_proxy_low_correlation_confirmed: `true`
- h1_teacher_architecture_too_weak: `true`
- h2_occlusion_not_faithful: `true`
- h3_frame_repeat_temporal_limitation: `true`
- h4_short_horizon_current_dominance: `false`
- h5_data_diversity_limited: `true`

```json
[
  "fixed-attention occlusion label is less useful than Step30A proxy topK on held-out sanity",
  "teacher architecture benefits from proxy-prior context scoring",
  "full context injects noise without selection",
  "frame-repeat VideoMAE tokens are still a temporal-representation limitation"
]
```
