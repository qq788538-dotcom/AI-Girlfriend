#!/bin/sh
set -eu

PROJECT_DIR="${VH_GPU_PROJECT_DIR:-/root/virtual-human}"
ENV_MODE="${VH_GPU_ENV_MODE:-venv}"
VENV_DIR="${VH_GPU_VENV_DIR:-$PROJECT_DIR/.venv-gpu}"
CONDA_ENV="${VH_GPU_CONDA_ENV:-flashhead}"
FLASHHEAD_DIR="$PROJECT_DIR/vendor/SoulX-FlashHead"
MODEL_DIR="$FLASHHEAD_DIR/models/SoulX-FlashHead-1_3B"
WAV2VEC_DIR="$FLASHHEAD_DIR/models/wav2vec2-base-960h"

run_python() {
    if [ "$ENV_MODE" = "venv" ]; then
        "$VENV_DIR/bin/python" "$@"
    else
        "$CONDA_BIN" run -n "$CONDA_ENV" python "$@"
    fi
}

run_tool() {
    tool_name="$1"
    shift
    if [ "$ENV_MODE" = "venv" ]; then
        "$VENV_DIR/bin/$tool_name" "$@"
    else
        "$CONDA_BIN" run -n "$CONDA_ENV" "$tool_name" "$@"
    fi
}

if [ "$(uname -s)" != "Linux" ]; then
    echo "This bootstrap script must run on the Linux GPU instance." >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; the instance is not ready for CUDA inference." >&2
    exit 1
fi
if [ ! -f "$FLASHHEAD_DIR/requirements.txt" ]; then
    echo "Project source is missing at $PROJECT_DIR. Upload or clone the workspace first." >&2
    exit 1
fi

nvidia-smi

if [ "$ENV_MODE" = "venv" ]; then
    if [ ! -x "$VENV_DIR/bin/python" ] || ! "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
        python3 -m venv --clear "$VENV_DIR"
    fi
elif [ "$ENV_MODE" = "conda" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BIN="$(command -v conda)"
    elif [ -x /root/miniconda3/bin/conda ]; then
        CONDA_BIN=/root/miniconda3/bin/conda
    else
        echo "conda is unavailable." >&2
        exit 1
    fi
    if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -Fx "$CONDA_ENV" >/dev/null 2>&1; then
        "$CONDA_BIN" create -y -n "$CONDA_ENV" python=3.10
    fi
else
    echo "VH_GPU_ENV_MODE must be venv or conda." >&2
    exit 1
fi

run_python -m pip install --upgrade pip wheel setuptools
run_python -m pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
# PyTorch's CUDA wheel owns the exact NCCL version. The upstream requirements
# currently pin a newer NCCL release that conflicts with torch 2.7.1+cu128.
FILTERED_REQUIREMENTS="$(mktemp)"
trap 'rm -f "$FILTERED_REQUIREMENTS"' EXIT
sed '/^nvidia-nccl-cu12==/d' \
    "$FLASHHEAD_DIR/requirements.txt" >"$FILTERED_REQUIREMENTS"
run_python -m pip install -r "$FILTERED_REQUIREMENTS"
run_python -m pip install ninja

export MAX_JOBS="${MAX_JOBS:-8}"
run_python -m pip install \
    flash_attn==2.8.0.post2 \
    --no-build-isolation

run_python -m pip install -e "$PROJECT_DIR"
run_python -m pip install "huggingface_hub[cli]"

if [ ! -f "$MODEL_DIR/Model_Lite/diffusion_pytorch_model.safetensors" ] || \
    [ ! -f "$MODEL_DIR/VAE_LTX/diffusion_pytorch_model.safetensors" ]; then
    run_tool hf download \
        Soul-AILab/SoulX-FlashHead-1_3B \
        --include "Model_Lite/*" \
        --include "VAE_LTX/*" \
        --local-dir "$MODEL_DIR"
fi
if [ ! -f "$WAV2VEC_DIR/config.json" ]; then
    run_tool hf download \
        facebook/wav2vec2-base-960h \
        --local-dir "$WAV2VEC_DIR"
fi

run_python -c \
    "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
run_python -c \
    "from virtual_human.flashhead_worker import FlashHeadSettings; print(FlashHeadSettings().model_type)"

echo "FlashHead GPU environment is ready."
echo "Start it with: VH_GPU_PROJECT_DIR=$PROJECT_DIR ./scripts/run-flashhead-gpu.sh"
