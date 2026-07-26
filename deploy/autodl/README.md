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
