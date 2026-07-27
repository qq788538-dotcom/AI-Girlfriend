# AutoDL 4090-48G deployment

This directory adapts `codex/xiangongyun-deploy` to AutoDL without changing the
Xiangongyun runtime. The model stack stays fully local:

- Qwen3.6-35B-A3B-AWQ on port 8000
- Qwen3-ASR-0.6B on port 8001
- Qwen3-Embedding-0.6B on port 8002
- Higgs Audio v3 on port 8010
- SoulX-FlashHead Lite on port 8770
- OpenViking on port 1934
- Public web and WebSocket gateway on port 6006

AutoDL maps container port 6006 to the instance HTTPS custom-service domain.
Only 6006 is public; every model service remains bound to loopback.

## Provision

Use an AutoDL Pro instance with one `4090-48G` (`v-48g`), CUDA driver support
at least 11.8, and at least 350 GB of expanded system disk.

```bash
git clone --recurse-submodules \
  https://github.com/qq788538-dotcom/AI-Girlfriend.git \
  /root/AI-Girlfriend
cd /root/AI-Girlfriend
git switch codex/autodl-deploy

mkdir -p runtime/voice-calibration
# Upload the locked reference file separately:
# runtime/voice-calibration/reference-female-only-complete-11s.wav

deploy/autodl/bootstrap.sh
deploy/autodl/control.sh start
deploy/autodl/control.sh status
```

The reference audio is intentionally excluded from Git. Bootstrap verifies its
fixed SHA-256 before installing the stack.

## Power-on command

Set the AutoDL Pro API `start_command` to:

```bash
bash /root/AI-Girlfriend/deploy/autodl/autostart.sh
```

The script is idempotent, waits for the GPU, writes
`runtime/autodl/autostart.log`, and does not start duplicate processes.

## SoulX-LiveAct test mode

For a single RTX 4090/5090, the optional LiveAct path follows the official
memory-saving flags (`--fp8_kv_cache --block_offload --t5_cpu`). The official
Flask demo is launched through `liveact_demo_launcher.py`, which forces its
listener to `127.0.0.1:5001`. On one GPU it runs directly without a distributed
rendezvous port. The protocol wrapper also remains loopback-only on
`127.0.0.1:8772`.

The control script defaults to the validated safe profile: PyTorch SDPA,
T5/reference conditioning caches, exact tail-frame generation, LightVAE
decode compilation disabled, the official 416x720 canvas
(`VH_LIVEACT_SIZE`), and PyTorch expandable CUDA allocator segments.
An installed SageAttention candidate is only enabled explicitly with
`VH_LIVEACT_FORCE_SDPA=0`; setting it back to `1` masks both FlashAttention
and SageAttention and provides a complete rollback path.

`VH_LIVEACT_BLOCK_OFFLOAD=1` remains the safe default for 32 GB cards. On
larger GPUs, set it to `0` to keep the 18B denoiser resident in VRAM and avoid
per-layer CPU/GPU transfers. Treat this as a hardware-specific A/B: validate
peak VRAM and the same media-quality gates before making it persistent.

```bash
deploy/autodl/liveact-control.sh start
deploy/autodl/liveact-control.sh status
deploy/autodl/liveact-control.sh logs
```

The web UI serves a pinned local HLS.js bundle for Chromium playback and keeps
the completed MP4 as a fallback. LiveAct model weights and generated videos
remain on the AutoDL data disk and are not committed to Git.
