#!/bin/sh
set -eu

script_dir="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
project_dir="${VH_XGC_PROJECT_DIR:-$(CDPATH= cd -- "$script_dir/../.." && pwd)}"

if [ "$(id -u)" -ne 0 ]; then
    echo "AutoDL bootstrap must run as root." >&2
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    git \
    jq \
    libsox-dev \
    openssl \
    python3-venv \
    rsync \
    screen \
    sox

if [ ! -f "$script_dir/runtime.env" ]; then
    cp "$script_dir/env.example" "$script_dir/runtime.env"
fi
if [ ! -f "$script_dir/secrets.env" ]; then
    cp "$script_dir/secrets.example" "$script_dir/secrets.env"
fi
chmod 600 "$script_dir/runtime.env" "$script_dir/secrets.env"

export VH_XGC_PROJECT_DIR="$project_dir"
export VH_MODEL_DOWNLOAD_ATTEMPTS="${VH_MODEL_DOWNLOAD_ATTEMPTS:-20}"
export VH_AUTODL_NETWORK_TURBO="${VH_AUTODL_NETWORK_TURBO:-1}"

# The PRO 6000 profile keeps speech and animation on the GPU but sends text
# generation to the configured external API. Do not install or download the
# 35B local LLM.
VH_BOOTSTRAP_LLM=false \
VH_BOOTSTRAP_MEMORY_LLM=true \
    "$project_dir/deploy/xiangongyun/bootstrap-models.sh"

bootstrap_venv="$project_dir/.venv-bootstrap"
gateway_venv="$project_dir/.venv-gpu"
uv="$bootstrap_venv/bin/uv"
if [ ! -x "$gateway_venv/bin/python" ]; then
    "$uv" venv --python 3.12 "$gateway_venv"
fi
"$uv" pip install --python "$gateway_venv/bin/python" -e "$project_dir"

credentials_dir="$project_dir/runtime/credentials"
renderer_token="$credentials_dir/avatar-renderer.token"
mkdir -p "$credentials_dir"
if [ ! -s "$renderer_token" ]; then
    umask 077
    openssl rand -hex 32 > "$renderer_token"
fi
chmod 600 "$renderer_token"

echo "AutoDL PRO 6000 cloud speech, memory, gateway, and LiveAct dependencies are ready."
echo "Run: deploy/autodl/control.sh start"
