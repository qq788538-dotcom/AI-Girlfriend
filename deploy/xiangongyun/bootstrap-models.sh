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
model_download_attempts="${VH_MODEL_DOWNLOAD_ATTEMPTS:-20}"
llm_enabled="${VH_BOOTSTRAP_LLM:-true}"
memory_llm_enabled="${VH_BOOTSTRAP_MEMORY_LLM:-false}"

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
if [ ! -x "$bootstrap_venv/bin/modelscope" ] || [ ! -x "$bootstrap_venv/bin/hf" ]; then
    "$uv" pip install \
        --python "$bootstrap_venv/bin/python" \
        modelscope \
        "huggingface_hub[cli]"
fi
modelscope="$bootstrap_venv/bin/modelscope"
"$uv" python install 3.12

snapshot_complete() {
    target="$1"
    marker="$2"
    marker_path="$target/$marker"
    [ -s "$marker_path" ] || return 1
    case "$marker" in
        *.index.json)
            "$bootstrap_venv/bin/python" -c '
import json
import pathlib
import sys

index = pathlib.Path(sys.argv[1])
payload = json.loads(index.read_text())
files = set(payload.get("weight_map", {}).values())
if not files or any(not (index.parent / name).is_file() or (index.parent / name).stat().st_size == 0 for name in files):
    raise SystemExit(1)
' "$marker_path"
            ;;
    esac
}

qwen36_awq_complete() {
    target="$1"
    snapshot_complete "$target" model.safetensors.index.json || return 1
    "$bootstrap_venv/bin/python" -c '
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
config = json.loads((root / "config.json").read_text())
quantization = config.get("quantization_config", {})
excluded = set(quantization.get("modules_to_not_convert", []))
required_exclusions = {"linear_attn", "self_attn", "shared_expert"}
if quantization.get("quant_method") != "awq" or not required_exclusions <= excluded:
    raise SystemExit(1)

weights = json.loads(
    (root / "model.safetensors.index.json").read_text()
).get("weight_map", {})
linear_weights = [
    name
    for name in weights
    if ".linear_attn." in name and name.endswith(".weight")
]
linear_qweights = [
    name
    for name in weights
    if ".linear_attn." in name and name.endswith(".qweight")
]
if len(linear_weights) < 100 or linear_qweights:
    raise SystemExit(1)
' "$target"
}

download_snapshot_complete() {
    target="$1"
    marker="$2"
    profile="$3"
    if [ "$profile" = qwen36_awq ]; then
        qwen36_awq_complete "$target"
    else
        snapshot_complete "$target" "$marker"
    fi
}

ensure_venv() {
    target="$1"
    if [ ! -x "$target/bin/python" ]; then
        "$uv" venv --python 3.12 "$target"
    fi
}

if [ "$llm_enabled" = true ]; then
    ensure_venv "$vllm_venv"
    if [ ! -x "$vllm_venv/bin/vllm" ]; then
        "$uv" pip install \
            --python "$vllm_venv/bin/python" \
            "vllm[audio]==$vllm_version"
    fi
    if [ ! -x "$vllm_venv/bin/ninja" ]; then
        "$uv" pip install \
            --python "$vllm_venv/bin/python" \
            ninja
    fi
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
if ! "$tts_venv/bin/python" -c \
    "import vllm, vllm_omni; assert vllm.__version__ == '$vllm_omni_version'" \
    >/dev/null 2>&1; then
    "$uv" pip install \
        --python "$tts_venv/bin/python" \
        "vllm==$vllm_omni_version" \
        --torch-backend=auto
    "$uv" pip install \
        --python "$tts_venv/bin/python" \
        "vllm-omni==$vllm_omni_version"
fi
if [ ! -x "$tts_venv/bin/ninja" ]; then
    "$uv" pip install \
        --python "$tts_venv/bin/python" \
        ninja
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
hf="$bootstrap_venv/bin/hf"
run_hf() {
    if [ "${VH_AUTODL_NETWORK_TURBO:-0}" = 1 ] && [ -f /etc/network_turbo ]; then
        (
            # AutoDL's proxy is useful for Hugging Face but intentionally not
            # exported to the rest of bootstrap, where it slows PyPI down.
            # shellcheck disable=SC1091
            . /etc/network_turbo >/dev/null
            exec "$hf" "$@"
        )
    else
        "$hf" "$@"
    fi
}
download_model() {
    repo="$1"
    target="$2"
    marker="$3"
    attempt=1
    while ! snapshot_complete "$target" "$marker"; do
        echo "Downloading $repo (attempt $attempt/$model_download_attempts)"
        if run_hf download "$repo" --local-dir "$target" &&
            snapshot_complete "$target" "$marker"; then
            break
        fi
        if [ "$attempt" -ge "$model_download_attempts" ]; then
            echo "Model download did not complete after $attempt attempts: $repo" >&2
            exit 1
        fi
        attempt=$((attempt + 1))
        sleep 5
    done
}

download_model_subset() {
    repo="$1"
    include_pattern="$2"
    target="$3"
    marker="$4"
    attempt=1
    while ! snapshot_complete "$target" "$marker"; do
        echo "Downloading $include_pattern from $repo (attempt $attempt/$model_download_attempts)"
        if run_hf download "$repo" \
            --include "$include_pattern" \
            --local-dir "$target" &&
            snapshot_complete "$target" "$marker"; then
            break
        fi
        if [ "$attempt" -ge "$model_download_attempts" ]; then
            echo "Model subset did not complete after $attempt attempts: $repo:$include_pattern" >&2
            exit 1
        fi
        attempt=$((attempt + 1))
        sleep 5
    done
}

download_modelscope_model() {
    repo="$1"
    target="$2"
    marker="$3"
    profile="${4:-generic}"
    attempt=1
    while ! download_snapshot_complete "$target" "$marker" "$profile"; do
        echo "Downloading $repo from ModelScope (attempt $attempt/$model_download_attempts)"
        if "$modelscope" download "$repo" \
            --local-dir "$target" \
            --max-workers 4 &&
            download_snapshot_complete "$target" "$marker" "$profile"
        then
            break
        fi
        if [ "$attempt" -ge "$model_download_attempts" ]; then
            echo "ModelScope download did not complete after $attempt attempts: $repo" >&2
            exit 1
        fi
        attempt=$((attempt + 1))
        sleep 5
    done
}

if [ "$llm_enabled" = true ]; then
    download_modelscope_model \
        tclf90/Qwen3.6-35B-A3B-AWQ \
        "$models_dir/Qwen3.6-35B-A3B-AWQ-QuantTrio" \
        model.safetensors.index.json \
        qwen36_awq
fi
if [ "$memory_llm_enabled" = true ]; then
    download_modelscope_model \
        Qwen/Qwen3-4B-AWQ \
        "$models_dir/Qwen3-4B-AWQ" \
        model.safetensors.index.json
fi
download_modelscope_model \
    Qwen/Qwen3-ASR-0.6B \
    "$models_dir/Qwen3-ASR-0.6B" \
    model.safetensors
download_model \
    bosonai/higgs-tts-3-4b \
    "$models_dir/higgs-tts-3-4b" \
    model.safetensors
download_model_subset \
    k2-fsa/OmniVoice \
    'audio_tokenizer/*' \
    "$models_dir/OmniVoice" \
    audio_tokenizer/model.safetensors
download_modelscope_model \
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
