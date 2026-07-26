#!/bin/bash
set -eu

project_dir="${VH_XGC_PROJECT_DIR:-/root/AI-Girlfriend}"
runtime_dir="$project_dir/runtime/autodl"
log_file="$runtime_dir/autostart.log"

mkdir -p "$runtime_dir"
exec >>"$log_file" 2>&1
echo "[$(date --iso-8601=seconds)] AutoDL autostart begin"

exec 9>/run/ai-girlfriend-autodl.lock
if ! flock -n 9; then
    echo "another AutoDL autostart process holds the lock"
    exit 0
fi

for attempt in $(seq 1 60); do
    if nvidia-smi >/dev/null 2>&1; then
        break
    fi
    if [ "$attempt" -eq 60 ]; then
        echo "GPU did not become ready"
        exit 1
    fi
    sleep 2
done

if [ ! -x "$project_dir/deploy/autodl/control.sh" ]; then
    echo "AutoDL deployment branch is not installed at $project_dir"
    exit 1
fi
if [ ! -f "$project_dir/deploy/autodl/runtime.env" ]; then
    echo "runtime.env is missing; run deploy/autodl/bootstrap.sh first"
    exit 1
fi

cd "$project_dir"
if deploy/autodl/control.sh status; then
    echo "all services are already healthy"
    exit 0
fi

deploy/autodl/control.sh start
deploy/autodl/control.sh status
echo "[$(date --iso-8601=seconds)] AutoDL autostart complete"
