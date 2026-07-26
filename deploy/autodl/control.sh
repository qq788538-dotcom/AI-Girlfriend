#!/bin/sh
set -eu

script_dir="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
project_dir="${VH_XGC_PROJECT_DIR:-$(CDPATH= cd -- "$script_dir/../.." && pwd)}"

export VH_XGC_PROJECT_DIR="$project_dir"
export VH_XGC_RUNTIME_ENV="${VH_AUTODL_RUNTIME_ENV:-$script_dir/runtime.env}"
export VH_XGC_SECRETS_ENV="${VH_AUTODL_SECRETS_ENV:-$script_dir/secrets.env}"

exec "$project_dir/deploy/xiangongyun/control.sh" "$@"
