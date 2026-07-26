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
apt-get install -y curl git jq openssl rsync screen

if [ ! -f "$script_dir/runtime.env" ]; then
    cp "$script_dir/env.example" "$script_dir/runtime.env"
fi
if [ ! -f "$script_dir/secrets.env" ]; then
    cp "$script_dir/secrets.example" "$script_dir/secrets.env"
fi
chmod 600 "$script_dir/runtime.env" "$script_dir/secrets.env"

export VH_XGC_PROJECT_DIR="$project_dir"
"$project_dir/deploy/xiangongyun/bootstrap.sh"

echo "AutoDL dependencies and models are ready."
echo "Run: deploy/autodl/control.sh start"
