# Minimal Pipeline Smoke Test Report

| Item | Value |
| --- | --- |
| Device | `cpu` |
| dummy video shape | `[2, 4, 3, 224, 224]` |
| full tokens shape | `[2, 196, 768]` |
| teacher output shape | `[2, 768]` |
| scores shape | `[2, 196]` |
| selected tokens shape | `[2, 40, 768]` |
| topk indices shape | `[2, 40]` |
| compressed latents shape | `[2, 16, 512]` |
| student output shape | `[2, 768]` |
| future loss | `0.05851075` |
| distill loss | `0.05851075` |
| budget loss | `0.14531241` |
| SMOKE_TEST_PASS | `true` |

SMOKE_TEST_PASS = true
