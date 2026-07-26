#!/bin/sh
set -eu

project_root="${VH_XGC_PROJECT_DIR:-$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)}"
venv="${VH_EMBEDDING_VENV_DIR:-$project_root/.venv-embedding}"
model_path="${VH_EMBEDDING_MODEL_PATH:-$project_root/runtime/models/Qwen3-Embedding-0.6B}"
host="${VH_EMBEDDING_HOST:-127.0.0.1}"
port="${VH_EMBEDDING_PORT:-8002}"

test -x "$venv/bin/virtual-human-embedding"
test -f "$model_path/model.safetensors"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${VH_EMBEDDING_CPU_THREADS:-8}"

exec "$venv/bin/virtual-human-embedding" \
    --model "$model_path" \
    --host "$host" \
    --port "$port" \
    --max-sequence-length "${VH_EMBEDDING_MAX_SEQUENCE_LENGTH:-2048}"
