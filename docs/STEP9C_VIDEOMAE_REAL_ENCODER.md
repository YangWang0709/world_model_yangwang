# STEP9C VideoMAE Real Frozen Encoder Report

## 1. Goal

Step 9C used one user-authorized VideoMAE checkpoint from the Windows local machine, transferred it to the Ubuntu host, and ran a real frozen encoder token extraction smoke on the existing `real_video_minimal` subset.

## 2. Cloud Server Decision

This stage did not require a cloud server. The real VideoMAE smoke succeeded on the local RTX 5080 server with conservative limits: `batch_size=1`, `num_workers=0`, `clip_len<=8`, `image_size=224`, and `max_samples=8`.

No CUDA OOM occurred. RAM stayed far below the 28 GiB stop threshold.

## 3. Download Scope

Allowed checkpoint: `MCG-NJU/videomae-base-finetuned-kinetics`.

Source policy: obtain the checkpoint on Windows local first. If absent, download it on Windows local, then transfer it to Ubuntu under ignored `model_cache/`. The Ubuntu smoke configs use `allow_download=false` and `local_files_only=true`.

Windows local source path:
`D:/world_model/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local`

Ubuntu local checkpoint path:
`/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local`

Transferred files:
- `config.json`
- `preprocessor_config.json`
- `model.safetensors`

Model SHA256:
`065cfde56f97671da4d196e1a3a6d3cf9304dcad92eb671e3632a6c6cdd04f73`

No V-JEPA weights were downloaded. No real large dataset was downloaded.

## 4. Files Added or Updated

- `.gitignore`
- `configs/videomae_real_video_smoke.yaml`
- `configs/token_extraction_real_video_videomae.yaml`
- `encoders/videomae_wrapper.py`
- `scripts/download_videomae_checkpoint.py`
- `scripts/smoke_test_videomae_real_encoder.py`
- `scripts/smoke_test_real_video_videomae_token_extraction.py`
- `scripts/extract_tokens.py`
- `tests/test_videomae_download_config.py`
- `tests/test_videomae_real_encoder_smoke_config.py`
- `docs/VIDEOMAE_DOWNLOAD_REPORT.md`
- `docs/VIDEOMAE_REAL_ENCODER_SMOKE_REPORT.md`
- `docs/STEP9C_VIDEOMAE_REAL_ENCODER.md`

## 5. VideoMAE Wrapper

- input: `[B,T,C,H,W]`
- output: `[B,N,D]`
- model is loaded with `eval()` and `requires_grad_(False)`
- CUDA OOM is caught and reported clearly
- external 4-frame clips are adapted to the configured 8-frame VideoMAE input without increasing dataset size
- `last_hidden_state` is used as token output

## 6. Encoder Smoke Result

See `docs/VIDEOMAE_REAL_ENCODER_SMOKE_REPORT.md`.

- model name/path: `/home/ubuntu22/tgpawb_world_model/model_cache/videomae-base-finetuned-kinetics-local`
- input shape: `[1, 8, 3, 224, 224]`
- output shape: `[1, 784, 768]`
- dtype: `torch.float32`
- GPU memory before/after: about `1.321 GiB` used -> `1.43 GiB` used
- RAM before/after: about `4.401 GiB` used -> `4.631 GiB` used
- OOM: `false`
- pass: `true`

## 7. Real Video Token Extraction Result

- samples used: `8`
- output dir: `/home/ubuntu22/tgpawb_world_model/data/token_shards/real_video_videomae_smoke`
- generated shard count: `4`
- first shard past/future shape: `[2, 784, 768]` / `[2, 784, 768]`
- encoder_name: `videomae`
- requested_encoder: `videomae`
- actual_encoder: `videomae`
- used_fallback: `false`

## 8. Pytest Result

`45 passed in 1.07s`.

Step 9B fallback smoke and Step 9C real VideoMAE smoke both passed after the code changes.

## 9. Git Commit

Branch: `feature/tgpawb-step3-token-extraction`.

Commit and push are performed after this report file is written. The final response records the exact commit hash and push status. No PR is created in this stage.

## 10. What Was Not Done

- no large dataset download
- no V-JEPA download
- no Teacher training
- no Student training
- no VLM grounding
- no RL / policy optimization
- no model weight committed
- no generated `.pt` committed
- no checkpoint committed
- no password or token saved
- no PR created

## 11. Next Step Recommendation

Step 10 can use the ignored `real_video_videomae_smoke` token shards for a tiny Teacher training smoke if authorized. A cloud 4090 or 48GB GPU server is not required for Step 9C based on this successful local RTX 5080 smoke.
