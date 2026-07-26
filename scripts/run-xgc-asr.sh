#!/bin/sh
set -eu

project_root="${VH_XGC_PROJECT_DIR:-$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)}"
venv="${VH_ASR_VENV_DIR:-$project_root/.venv-asr}"
model_path="${VH_ASR_MODEL_PATH:-$project_root/runtime/models/Qwen3-ASR-0.6B}"
host="${VH_ASR_HOST:-127.0.0.1}"
port="${VH_ASR_PORT:-8001}"

test -x "$venv/bin/virtual-human-asr"
test -f "$model_path/model.safetensors"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

exec "$venv/bin/virtual-human-asr" \
    --model "$model_path" \
    --host "$host" \
    --port "$port"
