#!/bin/sh
set -eu

project_root="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
runtime_dir="$project_root/runtime/openviking-girlfriend"
workspace_dir="$runtime_dir/workspace"
config_file="$runtime_dir/ov.conf"
log_file="$runtime_dir/server.log"
pid_file="$runtime_dir/server.pid"
port="${VH_MEMORY_PORT:-1934}"
openviking_bin="${OPENVIKING_SERVER_BIN:-/Users/james/.openviking/ov-server-venv/bin/openviking-server}"

mkdir -p "$runtime_dir" "$workspace_dir"

if [ -f "$project_root/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$project_root/.env"
    set +a
fi

write_config() {
    plan_key="${VH_ARK_API_KEY:-}"
    embedding_key="${VH_MEMORY_EMBEDDING_API_KEY:-}"
    if [ -z "$embedding_key" ] && [ -f "/Users/james/.openviking/ov.conf" ]; then
        embedding_key="$(jq -r '.embedding.dense.api_key // empty' /Users/james/.openviking/ov.conf)"
    fi
    if [ -z "$plan_key" ]; then
        echo "VH_ARK_API_KEY is required for OpenViking memory extraction" >&2
        exit 1
    fi
    if [ -z "$embedding_key" ]; then
        echo "VH_MEMORY_EMBEDDING_API_KEY is required for OpenViking vector search" >&2
        exit 1
    fi

    jq -n \
        --arg workspace "$workspace_dir" \
        --arg embedding_key "$embedding_key" \
        --arg plan_key "$plan_key" \
        '{
            storage: {
                workspace: $workspace,
                vectordb: {backend: "local"},
                agfs: {backend: "local"}
            },
            embedding: {
                max_concurrent: 10,
                max_retries: 3,
                dense: {
                    provider: "volcengine",
                    api_base: "https://ark.cn-beijing.volces.com/api/v3",
                    api_key: $embedding_key,
                    model: "doubao-embedding-vision-251215",
                    dimension: 1024,
                    input: "multimodal"
                }
            },
            vlm: {
                provider: "volcengine",
                api_base: "https://ark.cn-beijing.volces.com/api/plan/v3",
                api_key: $plan_key,
                model: "doubao-seed-2-0-pro-260215",
                thinking: false,
                max_concurrent: 8,
                max_retries: 3
            },
            memory: {version: "v2"},
            server: {auth_mode: "dev"}
        }' > "$config_file"
    chmod 600 "$config_file"
}

is_running() {
    [ -f "$pid_file" ] || return 1
    pid="$(cat "$pid_file")"
    kill -0 "$pid" 2>/dev/null
}

start_service() {
    if is_running; then
        echo "OpenViking virtual-girlfriend memory is already running (pid $(cat "$pid_file"))"
        exit 0
    fi
    write_config
    OPENVIKING_CONFIG_FILE="$config_file" nohup "$openviking_bin" \
        --config "$config_file" \
        --host 127.0.0.1 \
        --port "$port" \
        > "$log_file" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "$pid_file"
    attempts=0
    while [ "$attempts" -lt 30 ]; do
        if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            echo "OpenViking virtual-girlfriend memory ready at http://127.0.0.1:$port (pid $pid)"
            exit 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "OpenViking failed to start; inspect $log_file" >&2
            exit 1
        fi
        attempts=$((attempts + 1))
        sleep 1
    done
    echo "OpenViking did not become healthy within 30 seconds; inspect $log_file" >&2
    exit 1
}

stop_service() {
    if ! is_running; then
        echo "OpenViking virtual-girlfriend memory is not running"
        exit 0
    fi
    pid="$(cat "$pid_file")"
    kill "$pid"
    rm -f "$pid_file"
    echo "Stopped OpenViking virtual-girlfriend memory (pid $pid)"
}

status_service() {
    if is_running && curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
        echo "running pid=$(cat "$pid_file") url=http://127.0.0.1:$port workspace=$workspace_dir"
        exit 0
    fi
    echo "stopped url=http://127.0.0.1:$port workspace=$workspace_dir"
    exit 1
}

foreground_service() {
    write_config
    echo "$$" > "$pid_file"
    exec "$openviking_bin" \
        --config "$config_file" \
        --host 127.0.0.1 \
        --port "$port"
}

case "${1:-start}" in
    start) start_service ;;
    foreground) foreground_service ;;
    stop) stop_service ;;
    restart)
        stop_service || true
        start_service
        ;;
    status) status_service ;;
    config)
        write_config
        echo "$config_file"
        ;;
    *)
        echo "usage: $0 {start|foreground|stop|restart|status|config}" >&2
        exit 2
        ;;
esac
