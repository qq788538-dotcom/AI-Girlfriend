#!/bin/sh
set -eu

script_dir="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
project_dir="${VH_XGC_PROJECT_DIR:-$(CDPATH= cd -- "$script_dir/../.." && pwd)}"
bootstrap_venv="$project_dir/.venv-bootstrap"
vllm_venv="$project_dir/.venv-vllm"
asr_venv="$project_dir/.venv-asr"
tts_venv="$project_dir/.venv-vllm-omni"
embedding_venv="$project_dir/.venv-embedding"
memory_venv="$project_dir/.venv-openviking"
models_dir="$project_dir/runtime/models"
vllm_version="${VH_VLLM_VERSION:-0.19.1}"
vllm_omni_version="${VH_VLLM_OMNI_VERSION:-0.24.0}"
openviking_version="${OPENVIKING_VERSION:-0.4.11}"

if [ "$(uname -s)" != "Linux" ] || ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Local model bootstrap requires a Linux CUDA instance." >&2
    exit 1
fi
for command_name in python3 curl sha256sum; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "Missing required command: $command_name" >&2
        exit 1
    }
done

if [ ! -x "$bootstrap_venv/bin/uv" ]; then
    python3 -m venv "$bootstrap_venv"
    "$bootstrap_venv/bin/pip" install --upgrade pip uv
fi
uv="$bootstrap_venv/bin/uv"
"$uv" python install 3.12

ensure_venv() {
    target="$1"
    if [ ! -x "$target/bin/python" ]; then
        "$uv" venv --python 3.12 "$target"
    fi
}

ensure_venv "$vllm_venv"
if [ ! -x "$vllm_venv/bin/vllm" ]; then
    "$uv" pip install \
        --python "$vllm_venv/bin/python" \
        "vllm[audio]==$vllm_version" \
        "huggingface_hub[cli]"
fi

ensure_venv "$asr_venv"
if [ ! -x "$asr_venv/bin/virtual-human-asr" ]; then
    "$uv" pip install \
        --python "$asr_venv/bin/python" \
        qwen-asr \
        soundfile \
        -e "$project_dir"
fi

ensure_venv "$tts_venv"
if [ ! -x "$tts_venv/bin/vllm-omni" ]; then
    "$uv" pip install \
        --python "$tts_venv/bin/python" \
        "vllm-omni==$vllm_omni_version"
fi

ensure_venv "$embedding_venv"
if [ ! -x "$embedding_venv/bin/virtual-human-embedding" ]; then
    "$uv" pip install \
        --python "$embedding_venv/bin/python" \
        "sentence-transformers>=5,<6" \
        -e "$project_dir"
fi

ensure_venv "$memory_venv"
if [ ! -x "$memory_venv/bin/openviking-server" ]; then
    "$uv" pip install \
        --python "$memory_venv/bin/python" \
        "openviking==$openviking_version"
fi

mkdir -p "$models_dir"
hf="$vllm_venv/bin/hf"
download_model() {
    repo="$1"
    target="$2"
    marker="$3"
    if [ ! -f "$target/$marker" ]; then
        "$hf" download "$repo" --local-dir "$target"
    fi
}

download_model \
    mattbucci/Qwen3.6-35B-A3B-AWQ \
    "$models_dir/Qwen3.6-35B-A3B-AWQ" \
    model.safetensors.index.json
download_model \
    Qwen/Qwen3-ASR-0.6B \
    "$models_dir/Qwen3-ASR-0.6B" \
    model.safetensors
download_model \
    bosonai/higgs-tts-3-4b \
    "$models_dir/higgs-tts-3-4b" \
    model.safetensors
download_model \
    Qwen/Qwen3-Embedding-0.6B \
    "$models_dir/Qwen3-Embedding-0.6B" \
    model.safetensors

voice_reference="$project_dir/runtime/voice-calibration/reference-female-only-complete-11s.wav"
avatar_reference="$project_dir/public/avatar-ai-girlfriend-v6.png"
voice_sha="b01ccb2c0ad5427f1478219b96dfba0d27e12bad281a81da46fed470a83630d4"
avatar_sha="c8afa1d691711330e525db0048dd21e37ab71c46476a7504d0e75b4ae8fcd162"
test "$(sha256sum "$voice_reference" | awk '{print $1}')" = "$voice_sha"
test "$(sha256sum "$avatar_reference" | awk '{print $1}')" = "$avatar_sha"

echo "Xiangongyun local model environments and weights are ready."
