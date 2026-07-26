#!/bin/sh
set -eu

script_dir="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
project_dir="${VH_XGC_PROJECT_DIR:-$(CDPATH= cd -- "$script_dir/../.." && pwd)}"
voice_reference="$project_dir/runtime/voice-calibration/reference-female-only-complete-11s.wav"
voice_reference_sha256="b01ccb2c0ad5427f1478219b96dfba0d27e12bad281a81da46fed470a83630d4"

if [ "$(uname -s)" != "Linux" ]; then
    echo "Xiangongyun bootstrap must run on the Linux GPU instance." >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "CUDA GPU is unavailable." >&2
    exit 1
fi
for command_name in git python3 curl jq openssl sha256sum; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Missing required command: $command_name" >&2
        exit 1
    fi
done

cd "$project_dir"
git submodule update --init --recursive

if [ ! -f "$voice_reference" ]; then
    echo "Locked Higgs reference is missing: $voice_reference" >&2
    echo "Upload it through SSH; it is intentionally excluded from the public Git repository." >&2
    exit 1
fi
actual_voice_sha256="$(sha256sum "$voice_reference" | awk '{print $1}')"
if [ "$actual_voice_sha256" != "$voice_reference_sha256" ]; then
    echo "Locked Higgs reference SHA256 mismatch; refusing to continue." >&2
    exit 1
fi

export VH_GPU_PROJECT_DIR="$project_dir"
"$project_dir/scripts/bootstrap-flashhead-gpu.sh"

"$script_dir/bootstrap-models.sh"

credentials_dir="$project_dir/runtime/credentials"
renderer_token="$credentials_dir/avatar-renderer.token"
mkdir -p "$credentials_dir"
if [ ! -s "$renderer_token" ]; then
    umask 077
    openssl rand -hex 32 > "$renderer_token"
fi
chmod 600 "$renderer_token"

echo "Xiangongyun dependencies are ready."
echo "Copy env.example to runtime.env and secrets.example to secrets.env, then run control.sh start."
