#!/bin/sh
set -eu

PROJECT_DIR="${VH_GPU_PROJECT_DIR:-/root/virtual-human}"
ENV_MODE="${VH_GPU_ENV_MODE:-venv}"
VENV_DIR="${VH_GPU_VENV_DIR:-$PROJECT_DIR/.venv-gpu}"
CONDA_ENV="${VH_GPU_CONDA_ENV:-flashhead}"
FLASHHEAD_DIR="$PROJECT_DIR/vendor/SoulX-FlashHead"
MODEL_TYPE="${VH_FLASHHEAD_MODEL_TYPE:-lite}"
TRANSPORT="${VH_FLASHHEAD_TRANSPORT:-mse}"
HOST="${VH_FLASHHEAD_HOST:-127.0.0.1}"
PORT="${VH_FLASHHEAD_PORT:-8770}"

if [ "$MODEL_TYPE" != "lite" ] && [ "${VH_FLASHHEAD_ALLOW_NON_REALTIME:-0}" != "1" ]; then
    echo "Refusing to start FlashHead $MODEL_TYPE as the realtime production worker." >&2
    echo "Use Lite, or set VH_FLASHHEAD_ALLOW_NON_REALTIME=1 for an explicit offline quality test." >&2
    exit 2
fi

if command -v ss >/dev/null 2>&1 &&
    ss -H -ltn "sport = :$PORT" 2>/dev/null | grep -q .; then
    echo "Refusing to start a duplicate FlashHead worker: $HOST:$PORT is already listening." >&2
    exit 3
fi

export VH_FLASHHEAD_REPO_DIR="$FLASHHEAD_DIR"
export VH_FLASHHEAD_CKPT_DIR="$FLASHHEAD_DIR/models/SoulX-FlashHead-1_3B"
export VH_FLASHHEAD_WAV2VEC_DIR="$FLASHHEAD_DIR/models/wav2vec2-base-960h"
export VH_FLASHHEAD_MODEL_TYPE="$MODEL_TYPE"
export VH_FLASHHEAD_TRANSPORT="$TRANSPORT"
export VH_FLASHHEAD_VIDEO_CRF="${VH_FLASHHEAD_VIDEO_CRF:-18}"
export VH_FLASHHEAD_VIDEO_SHARPEN="${VH_FLASHHEAD_VIDEO_SHARPEN:-true}"
export VH_FLASHHEAD_OUTPUT_SIZE="${VH_FLASHHEAD_OUTPUT_SIZE:-768}"
export VH_FLASHHEAD_REFERENCE_LOCKED="${VH_FLASHHEAD_REFERENCE_LOCKED:-true}"
export VH_FLASHHEAD_REFERENCE_SHA256="${VH_FLASHHEAD_REFERENCE_SHA256:-c8afa1d691711330e525db0048dd21e37ab71c46476a7504d0e75b4ae8fcd162}"
export VH_FLASHHEAD_HOST="$HOST"
export VH_FLASHHEAD_PORT="$PORT"
export VH_FLASHHEAD_RUNTIME_DIR="${VH_FLASHHEAD_RUNTIME_DIR:-$PROJECT_DIR/runtime/flashhead}"
export VH_FLASHHEAD_ACCESS_TOKEN_FILE="${VH_FLASHHEAD_ACCESS_TOKEN_FILE:-$PROJECT_DIR/runtime/credentials/avatar-renderer.token}"
export FLASHHEAD_COMPILE_MODEL="${FLASHHEAD_COMPILE_MODEL:-1}"
export FLASHHEAD_COMPILE_VAE="${FLASHHEAD_COMPILE_VAE:-1}"
# Keep generated TorchInductor kernels on the persistent project disk. Both
# Lite and Pro have a long first compile on Ada GPUs, while cached steady-state
# inference is much faster.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PROJECT_DIR/runtime/torchinductor-cache}"
# FlashHead already loads the transformer and VAE as BF16. A 48 GB 4090 D has
# enough room to keep them resident, so avoid quality-reducing quantization or
# CPU offload and use the allocator setting intended for long-running services.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
# FlashHead loads lazily from a WebSocket worker thread. A multi-process
# TorchInductor compiler pool can deadlock when it forks from that thread.
# Serial compilation is only a cold-start cost; cached inference is unaffected.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
cd "$PROJECT_DIR"
if [ "$ENV_MODE" = "venv" ]; then
    exec "$VENV_DIR/bin/python" -m virtual_human.flashhead_worker
fi
if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
elif [ -x /root/miniconda3/bin/conda ]; then
    CONDA_BIN=/root/miniconda3/bin/conda
else
    echo "conda is unavailable." >&2
    exit 1
fi
exec "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" python -m virtual_human.flashhead_worker
