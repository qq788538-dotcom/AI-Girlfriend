#!/bin/sh
set -eu

project_root="${VH_XGC_PROJECT_DIR:-$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)}"
venv="${VH_LLM_VENV_DIR:-$project_root/.venv-vllm}"
model_path="${VH_LLM_MODEL_PATH:-$project_root/runtime/models/Qwen3.6-35B-A3B-AWQ}"
served_name="${VH_OMLX_CHAT_MODEL:-Qwen3.6-35B-A3B-AWQ}"
host="${VH_LLM_HOST:-127.0.0.1}"
port="${VH_LLM_PORT:-8000}"
gpu_utilization="${VH_LLM_GPU_MEMORY_UTILIZATION:-0.455}"
max_model_len="${VH_LLM_MAX_MODEL_LEN:-4096}"

test -x "$venv/bin/vllm"
"$venv/bin/python" -c '
import json
import pathlib
import sys

index = pathlib.Path(sys.argv[1])
payload = json.loads(index.read_text())
files = set(payload.get("weight_map", {}).values())
if not files or any(not (index.parent / name).is_file() or (index.parent / name).stat().st_size == 0 for name in files):
    raise SystemExit("LLM snapshot is incomplete")
' "$model_path/model.safetensors.index.json"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

exec "$venv/bin/vllm" serve "$model_path" \
    --served-model-name "$served_name" \
    --host "$host" \
    --port "$port" \
    --dtype bfloat16 \
    --gpu-memory-utilization "$gpu_utilization" \
    --max-model-len "$max_model_len" \
    --max-num-seqs 2 \
    --kv-cache-dtype fp8 \
    --language-model-only \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code
