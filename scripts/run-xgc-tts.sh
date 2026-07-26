#!/bin/sh
set -eu

project_root="${VH_XGC_PROJECT_DIR:-$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)}"
venv="${VH_TTS_VENV_DIR:-$project_root/.venv-vllm-omni}"
model_path="${VH_TTS_MODEL_PATH:-$project_root/runtime/models/higgs-tts-3-4b}"
host="${VH_TTS_HOST:-127.0.0.1}"
port="${VH_TTS_PORT:-8010}"
stage_overrides="${VH_TTS_STAGE_OVERRIDES:-{\"0\":{\"gpu_memory_utilization\":0.27,\"max_num_seqs\":1},\"1\":{\"gpu_memory_utilization\":0.05,\"max_num_seqs\":1}}}"

test -x "$venv/bin/vllm-omni"
test -f "$model_path/model.safetensors"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

exec "$venv/bin/vllm-omni" serve "$model_path" \
    --host "$host" \
    --port "$port" \
    --trust-remote-code \
    --omni \
    --stage-overrides "$stage_overrides"
