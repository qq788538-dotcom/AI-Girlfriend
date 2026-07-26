#!/bin/sh
set -eu

script_dir="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
project_dir="${VH_XGC_PROJECT_DIR:-$(CDPATH= cd -- "$script_dir/../.." && pwd)}"
runtime_env="${VH_XGC_RUNTIME_ENV:-$script_dir/runtime.env}"
secrets_env="${VH_XGC_SECRETS_ENV:-$script_dir/secrets.env}"
service_dir="$project_dir/runtime/xiangongyun"
gateway_pid="$service_dir/gateway.pid"
renderer_pid="$service_dir/flashhead.pid"
gateway_log="$service_dir/gateway.log"
renderer_log="$service_dir/flashhead.log"

load_environment() {
    if [ ! -f "$runtime_env" ]; then
        echo "Missing $runtime_env. Copy env.example to runtime.env first." >&2
        exit 1
    fi
    if [ ! -f "$secrets_env" ]; then
        echo "Missing $secrets_env. Copy secrets.example to secrets.env first." >&2
        exit 1
    fi
    set -a
    # shellcheck disable=SC1090
    . "$runtime_env"
    # shellcheck disable=SC1090
    . "$secrets_env"
    set +a

    export VH_XGC_PROJECT_DIR="$project_dir"
    export VH_GPU_PROJECT_DIR="$project_dir"
    export VH_HOST="${VH_HOST:-0.0.0.0}"
    export VH_PORT="${VH_PORT:-80}"
    export VH_PUBLIC_DIR="${VH_PUBLIC_DIR:-$project_dir/public}"
    export VH_AVATAR_RENDERER_WS="${VH_AVATAR_RENDERER_WS:-ws://127.0.0.1:8770/avatar}"
    export VH_AVATAR_RENDERER_TOKEN_FILE="$project_dir/runtime/credentials/avatar-renderer.token"
    export VH_FLASHHEAD_HOST="${VH_FLASHHEAD_HOST:-127.0.0.1}"
    export VH_FLASHHEAD_PORT="${VH_FLASHHEAD_PORT:-8770}"
    export VH_FLASHHEAD_ACCESS_TOKEN_FILE="$VH_AVATAR_RENDERER_TOKEN_FILE"
    export VH_MEMORY_BASE_URL="${VH_MEMORY_BASE_URL:-http://127.0.0.1:1934}"
    export VH_MEMORY_PORT="${VH_MEMORY_PORT:-1934}"
    export OPENVIKING_SERVER_BIN="${OPENVIKING_SERVER_BIN:-$project_dir/.venv-openviking/bin/openviking-server}"
}

is_running() {
    pid_file="$1"
    [ -f "$pid_file" ] || return 1
    pid="$(cat "$pid_file")"
    kill -0 "$pid" 2>/dev/null
}

stop_pid() {
    name="$1"
    pid_file="$2"
    if ! is_running "$pid_file"; then
        rm -f "$pid_file"
        echo "$name: stopped"
        return
    fi
    pid="$(cat "$pid_file")"
    kill "$pid"
    attempts=0
    while kill -0 "$pid" 2>/dev/null && [ "$attempts" -lt 20 ]; do
        attempts=$((attempts + 1))
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "$name did not stop cleanly (pid $pid)." >&2
        return 1
    fi
    rm -f "$pid_file"
    echo "$name: stopped"
}

wait_for_url() {
    name="$1"
    url="$2"
    pid_file="$3"
    attempts=0
    while [ "$attempts" -lt 120 ]; do
        if curl -fsS "$url" >/dev/null 2>&1; then
            echo "$name: ready"
            return
        fi
        if [ -n "$pid_file" ] && ! is_running "$pid_file"; then
            echo "$name exited before becoming ready." >&2
            exit 1
        fi
        attempts=$((attempts + 1))
        sleep 1
    done
    echo "$name did not become ready: $url" >&2
    exit 1
}

validate_start() {
    if [ ! -x "$project_dir/.venv-gpu/bin/virtual-human" ]; then
        echo "GPU environment is missing. Run bootstrap.sh first." >&2
        exit 1
    fi
    if [ ! -x "$OPENVIKING_SERVER_BIN" ]; then
        echo "OpenViking environment is missing. Run bootstrap.sh first." >&2
        exit 1
    fi
    if [ ! -s "$VH_AVATAR_RENDERER_TOKEN_FILE" ]; then
        echo "Renderer token is missing. Run bootstrap.sh first." >&2
        exit 1
    fi
    if [ "${VH_MEMORY_ENABLED:-false}" = "true" ]; then
        if [ -z "${VH_ARK_API_KEY:-}" ] || [ -z "${VH_MEMORY_EMBEDDING_API_KEY:-}" ]; then
            echo "OpenViking requires VH_ARK_API_KEY and VH_MEMORY_EMBEDDING_API_KEY." >&2
            exit 1
        fi
    fi
    if [ "${VH_UPSTREAM_MODE:-omlx}" = "omlx" ]; then
        case "${VH_OMLX_BASE_URL:-}" in
            http://127.0.0.1*|http://localhost*|"")
                echo "Cloud oMLX URL must point to the approved Mac endpoint, not localhost." >&2
                exit 1
                ;;
        esac
        case "${VH_TTS_BASE_URL:-}" in
            http://127.0.0.1*|http://localhost*|"")
                echo "Cloud Higgs URL must point to the approved Mac endpoint, not localhost." >&2
                exit 1
                ;;
        esac
    fi
}

start_services() {
    validate_start
    mkdir -p "$service_dir"
    cd "$project_dir"

    if [ "${VH_MEMORY_ENABLED:-false}" = "true" ]; then
        "$project_dir/scripts/run-openviking-memory.sh" start
    fi

    if ! is_running "$renderer_pid"; then
        nohup "$project_dir/scripts/run-flashhead-gpu.sh" > "$renderer_log" 2>&1 < /dev/null &
        echo "$!" > "$renderer_pid"
    fi
    wait_for_url "FlashHead" "http://127.0.0.1:${VH_FLASHHEAD_PORT}/healthz" "$renderer_pid"

    if ! is_running "$gateway_pid"; then
        nohup "$project_dir/.venv-gpu/bin/virtual-human" > "$gateway_log" 2>&1 < /dev/null &
        echo "$!" > "$gateway_pid"
    fi
    wait_for_url "gateway" "http://127.0.0.1:${VH_PORT}/healthz" "$gateway_pid"
    echo "Public entry: use the Xiangongyun instance web URL."
}

stop_services() {
    stop_pid "gateway" "$gateway_pid"
    stop_pid "FlashHead" "$renderer_pid"
    if [ "${VH_MEMORY_ENABLED:-false}" = "true" ]; then
        "$project_dir/scripts/run-openviking-memory.sh" stop
    fi
}

status_services() {
    exit_code=0
    if is_running "$gateway_pid" && curl -fsS "http://127.0.0.1:${VH_PORT}/healthz" >/dev/null 2>&1; then
        echo "gateway: running"
    else
        echo "gateway: stopped"
        exit_code=1
    fi
    if is_running "$renderer_pid" && curl -fsS "http://127.0.0.1:${VH_FLASHHEAD_PORT}/healthz" >/dev/null 2>&1; then
        echo "FlashHead: running"
    else
        echo "FlashHead: stopped"
        exit_code=1
    fi
    if [ "${VH_MEMORY_ENABLED:-false}" = "true" ]; then
        if "$project_dir/scripts/run-openviking-memory.sh" status; then
            :
        else
            exit_code=1
        fi
    else
        echo "OpenViking: disabled"
    fi
    return "$exit_code"
}

load_environment
case "${1:-status}" in
    start) start_services ;;
    stop) stop_services ;;
    restart)
        stop_services || true
        start_services
        ;;
    status) status_services ;;
    logs)
        echo "gateway: $gateway_log"
        echo "FlashHead: $renderer_log"
        echo "OpenViking: $project_dir/runtime/openviking-girlfriend/server.log"
        ;;
    *)
        echo "usage: $0 {start|stop|restart|status|logs}" >&2
        exit 2
        ;;
esac
