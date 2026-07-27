# AutoDL PRO 6000 all-cloud deployment

This profile runs the complete browser-to-avatar path on one RTX PRO 6000
Blackwell 96 GB. The browser only records microphone PCM and plays HLS/MP4:

- Qwen3-4B-AWQ dialogue and private memory extraction on port 8000
- Qwen3-ASR-0.6B on port 8001
- Qwen3-Embedding-0.6B on port 8002
- Higgs Audio v3 on port 8010
- SoulX-LiveAct on port 5001 with its protocol wrapper on port 8772
- OpenViking on port 1934
- Public web and WebSocket gateway on port 6006

AutoDL maps container port 6006 to the instance HTTPS custom-service domain.
Only 6006 is public; every model, memory, and renderer service is loopback-only.
Silero VAD v6.2 runs locally in the browser so silence is not uploaded.

## Locked best-value profile

The cost/performance deployment stays on one RTX PRO 6000: Qwen3-4B-AWQ for
dialogue and memory extraction, local Qwen3 ASR/embedding, local Higgs TTS,
and SoulX-LiveAct at 512x512/20 FPS with resident weights, SageAttention2,
cached conditioning, and static LightVAE decode compilation. It does not use
FP4/SageAttention3, sparse attention, lower resolution, lower frame rate, or a
second GPU. `runtime.env` and `env.example` carry the same reproducible profile.

## Provision

Use an AutoDL Pro instance with one RTX PRO 6000 Blackwell 96 GB, CUDA 12.8,
and at least 50 GB free on the persistent data disk after LiveAct is installed.

```bash
git clone --recurse-submodules \
  https://github.com/qq788538-dotcom/AI-Girlfriend.git \
  /root/AI-Girlfriend
cd /root/AI-Girlfriend
git switch codex/autodl-cloud-fullstack

mkdir -p runtime/voice-calibration
# Upload the locked reference file separately:
# runtime/voice-calibration/reference-female-only-complete-11s.wav

deploy/autodl/bootstrap.sh
deploy/autodl/control.sh start
deploy/autodl/control.sh status
```

The reference audio is intentionally excluded from Git. Bootstrap verifies its
fixed SHA-256 before installing the stack.

The bootstrap uses ModelScope for Qwen checkpoints and enables AutoDL network
turbo only around Hugging Face downloads. It deliberately skips the 35B model;
the compact AWQ model leaves the validated resident LiveAct profile and Higgs
TTS enough VRAM to run concurrently.

## Migrate OpenViking memory

Stop the local OpenViking writer, copy the workspace without printing its
contents, and verify file count and byte count before starting cloud services:

```bash
rsync -a --info=progress2 \
  runtime/openviking-girlfriend/workspace/ \
  root@INSTANCE:/root/AI-Girlfriend/runtime/openviking-girlfriend/workspace/
```

The deployment process makes a timestamped cloud-side backup before replacing
an existing workspace. OpenViking uses the loopback CPU embedding service and
the loopback 4B LLM; private memory extraction is not sent to an external API.

## Power-on command

Set the AutoDL Pro API `start_command` to:

```bash
bash /root/AI-Girlfriend/deploy/autodl/autostart.sh
```

The script is idempotent, waits for the GPU, writes
`runtime/autodl/autostart.log`, and does not start duplicate processes.

## SoulX-LiveAct runtime

The official
Flask demo is launched through `liveact_demo_launcher.py`, which forces its
listener to `127.0.0.1:5001`. On one GPU it runs directly without a distributed
rendezvous port. The protocol wrapper also remains loopback-only on
`127.0.0.1:8772`.

The control script defaults to the validated safe profile: T5/reference
conditioning caches, exact tail-frame generation, static LightVAE decode
compilation, the official 512x512 canvas (`VH_LIVEACT_SIZE`), persistent
TorchInductor graph caching, and PyTorch expandable CUDA allocator segments.
Static VAE compilation changes neither resolution nor generated frame count;
its fixed-input A/B gate checks media continuity, decoded-frame PSNR/SSIM,
audio identity, and mouth-motion timing before deployment.

An installed SageAttention candidate is enabled explicitly with
`VH_LIVEACT_FORCE_SDPA=0`; setting it back to `1` masks both FlashAttention
and SageAttention and provides a complete rollback path. The deployed
precision-sensitive profile stays on SageAttention2 rather than the FP4
SageAttention3 path.

The PRO 6000 profile uses `VH_LIVEACT_BLOCK_OFFLOAD=0` and SageAttention because
the measured resident path is faster and fits alongside the compact cloud
speech stack. The rollback values are `VH_LIVEACT_BLOCK_OFFLOAD=1` and
`VH_LIVEACT_FORCE_SDPA=1`.

The first process start compiles the denoiser and LightVAE graphs. Compiled
artifacts live in `/root/.cache/torchinductor-liveact`, so later restarts reuse
the same-GPU cache. Normal inference performance begins after the production
reference warm-up has completed.

```bash
deploy/autodl/liveact-control.sh start
deploy/autodl/liveact-control.sh status
deploy/autodl/liveact-control.sh logs
```

The web UI serves a pinned local HLS.js bundle for Chromium playback and keeps
the completed MP4 as a fallback. LiveAct model weights and generated videos
remain on the AutoDL data disk and are not committed to Git.
