#!/bin/sh
set -eu

project_root="${VH_XGC_PROJECT_DIR:-$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)}"
venv="${VH_LLM_VENV_DIR:-$project_root/.venv-vllm}"
model_path="${VH_LLM_MODEL_PATH:-$project_root/runtime/models/Qwen3.6-35B-A3B-AWQ-QuantTrio}"
served_name="${VH_OMLX_CHAT_MODEL:-Qwen3.6-35B-A3B-AWQ}"
host="${VH_LLM_HOST:-127.0.0.1}"
port="${VH_LLM_PORT:-8000}"
gpu_utilization="${VH_LLM_GPU_MEMORY_UTILIZATION:-0.48}"
max_model_len="${VH_LLM_MAX_MODEL_LEN:-8192}"
max_num_batched_tokens="${VH_LLM_MAX_NUM_BATCHED_TOKENS:-4096}"
attention_backend="${VH_LLM_ATTENTION_BACKEND:-FLASH_ATTN}"
kv_cache_dtype="${VH_LLM_KV_CACHE_DTYPE:-auto}"
quantization="${VH_LLM_QUANTIZATION:-awq}"
dtype="${VH_LLM_DTYPE:-float16}"
tool_call_parser="${VH_LLM_TOOL_CALL_PARSER:-qwen3_coder}"

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
export PATH="$venv/bin:$PATH"

exec "$venv/bin/vllm" serve "$model_path" \
    --served-model-name "$served_name" \
    --host "$host" \
    --port "$port" \
    --dtype "$dtype" \
    --quantization "$quantization" \
    --gpu-memory-utilization "$gpu_utilization" \
    --max-model-len "$max_model_len" \
    --max-num-batched-tokens "$max_num_batched_tokens" \
    --attention-backend "$attention_backend" \
    --max-num-seqs 2 \
    --kv-cache-dtype "$kv_cache_dtype" \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser "$tool_call_parser" \
    --enable-prefix-caching \
    --enforce-eager \
    --trust-remote-code
