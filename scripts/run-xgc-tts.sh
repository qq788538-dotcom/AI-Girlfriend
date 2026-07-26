#!/bin/sh
set -eu

project_root="${VH_XGC_PROJECT_DIR:-$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)}"
venv="${VH_TTS_VENV_DIR:-$project_root/.venv-vllm-omni}"
model_path="${VH_TTS_MODEL_PATH:-$project_root/runtime/models/higgs-tts-3-4b}"
served_name="${VH_TTS_SERVED_MODEL:-higgs_audio_v3}"
audio_tokenizer_path="${HIGGS_AUDIO_TOKENIZER_PATH:-$project_root/runtime/models/OmniVoice/audio_tokenizer}"
host="${VH_TTS_HOST:-127.0.0.1}"
port="${VH_TTS_PORT:-8010}"
stage_overrides="${VH_TTS_STAGE_OVERRIDES:-}"
if [ -z "$stage_overrides" ]; then
    stage_overrides='{"0":{"gpu_memory_utilization":0.27,"max_num_seqs":1,"attention_backend":"FLASH_ATTN"},"1":{"gpu_memory_utilization":0.05,"max_num_seqs":1,"attention_backend":"FLASH_ATTN"}}'
fi

test -x "$venv/bin/vllm"
test -s "$model_path/model.safetensors"
test -s "$audio_tokenizer_path/model.safetensors"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Xiangongyun mounts CUDA's cicc helper without execute permission. The
# FlashInfer sampler tries to JIT-compile on first use and cannot succeed on
# that mount, so use vLLM's built-in sampler implementation.
export VLLM_USE_FLASHINFER_SAMPLER=0
export HIGGS_AUDIO_TOKENIZER_PATH="$audio_tokenizer_path"
export PATH="$venv/bin:$PATH"

exec "$venv/bin/vllm" serve "$model_path" \
    --served-model-name "$served_name" \
    --host "$host" \
    --port "$port" \
    --trust-remote-code \
    --omni \
    --stage-overrides "$stage_overrides"
