#!/bin/sh
set -eu

script_dir="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
project_dir="${VH_XGC_PROJECT_DIR:-$(CDPATH= cd -- "$script_dir/../.." && pwd)}"
runtime_env="${VH_XGC_RUNTIME_ENV:-$script_dir/runtime.env}"
secrets_env="${VH_XGC_SECRETS_ENV:-$script_dir/secrets.env}"
service_dir="$project_dir/runtime/xiangongyun"

embedding_pid="$service_dir/embedding.pid"
llm_pid="$service_dir/llm.pid"
asr_pid="$service_dir/asr.pid"
tts_pid="$service_dir/tts.pid"
renderer_pid="$service_dir/flashhead.pid"
gateway_pid="$service_dir/gateway.pid"

embedding_log="$service_dir/embedding.log"
llm_log="$service_dir/llm.log"
asr_log="$service_dir/asr.log"
tts_log="$service_dir/tts.log"
renderer_log="$service_dir/flashhead.log"
gateway_log="$service_dir/gateway.log"

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
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1

    # A parent shell may contain cloud credentials. The offline deployment
    # explicitly removes them so no library fallback can reach a provider.
    unset OPENAI_API_KEY VH_ARK_API_KEY VH_MEMORY_EMBEDDING_API_KEY
}

is_running() {
    pid_file="$1"
    [ -f "$pid_file" ] || return 1
    pid="$(cat "$pid_file")"
    kill -0 "$pid" 2>/dev/null
}

start_process() {
    name="$1"
    pid_file="$2"
    log_file="$3"
    shift 3
    if is_running "$pid_file"; then
        echo "$name: already running (pid $(cat "$pid_file"))"
        return
    fi
    rm -f "$pid_file"
    nohup "$@" > "$log_file" 2>&1 < /dev/null &
    echo "$!" > "$pid_file"
    echo "$name: starting (pid $(cat "$pid_file"))"
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
    while kill -0 "$pid" 2>/dev/null && [ "$attempts" -lt 30 ]; do
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
    max_attempts="${4:-300}"
    attempts=0
    while [ "$attempts" -lt "$max_attempts" ]; do
        if curl -fsS "$url" >/dev/null 2>&1; then
            echo "$name: ready"
            return
        fi
        if ! is_running "$pid_file"; then
            echo "$name exited before becoming ready; inspect ${name}.log." >&2
            exit 1
        fi
        attempts=$((attempts + 1))
        sleep 1
    done
    echo "$name did not become ready: $url" >&2
    exit 1
}

require_loopback() {
    name="$1"
    endpoint="$2"
    case "$endpoint" in
        http://127.0.0.1/*|http://127.0.0.1:*|http://localhost/*|http://localhost:*|\
        ws://127.0.0.1/*|ws://127.0.0.1:*|ws://localhost/*|ws://localhost:*) ;;
        *)
            echo "Offline runtime requires loopback-only $name: $endpoint" >&2
            exit 1
            ;;
    esac
}

validate_start() {
    if [ "${VH_OFFLINE_RUNTIME:-false}" != "true" ]; then
        echo "VH_OFFLINE_RUNTIME=true is mandatory on Xiangongyun." >&2
        exit 1
    fi
    if [ "${VH_UPSTREAM_MODE:-}" != "omlx" ] || [ "${VH_CHAT_BACKEND:-}" != "omlx" ]; then
        echo "Offline runtime requires omlx upstream and chat backends." >&2
        exit 1
    fi
    if [ "${VH_MEMORY_LOCAL_ONLY:-false}" != "true" ]; then
        echo "OpenViking must use VH_MEMORY_LOCAL_ONLY=true." >&2
        exit 1
    fi
    if [ "${VH_TTS_PROTOCOL:-}" != "vllm_omni_higgs" ]; then
        echo "Xiangongyun Higgs must use VH_TTS_PROTOCOL=vllm_omni_higgs." >&2
        exit 1
    fi

    require_loopback "VH_OMLX_BASE_URL" "${VH_OMLX_BASE_URL:-}"
    require_loopback "VH_ASR_BASE_URL" "${VH_ASR_BASE_URL:-}"
    require_loopback "VH_TTS_BASE_URL" "${VH_TTS_BASE_URL:-}"
    require_loopback "VH_MEMORY_BASE_URL" "${VH_MEMORY_BASE_URL:-}"
    require_loopback "VH_AVATAR_RENDERER_WS" "${VH_AVATAR_RENDERER_WS:-}"
    require_loopback "VH_MEMORY_EMBEDDING_BASE_URL" "${VH_MEMORY_EMBEDDING_BASE_URL:-}"
    require_loopback "VH_MEMORY_VLM_BASE_URL" "${VH_MEMORY_VLM_BASE_URL:-}"

    for executable in \
        "$project_dir/.venv-gpu/bin/virtual-human" \
        "$project_dir/.venv-vllm/bin/vllm" \
        "$project_dir/.venv-asr/bin/virtual-human-asr" \
        "$project_dir/.venv-vllm-omni/bin/vllm-omni" \
        "$project_dir/.venv-embedding/bin/virtual-human-embedding" \
        "$OPENVIKING_SERVER_BIN"; do
        if [ ! -x "$executable" ]; then
            echo "Missing runtime executable: $executable" >&2
            echo "Run bootstrap.sh first." >&2
            exit 1
        fi
    done

    if [ ! -s "$VH_AVATAR_RENDERER_TOKEN_FILE" ]; then
        echo "Renderer token is missing. Run bootstrap.sh first." >&2
        exit 1
    fi
    test "$(sha256sum "$VH_OMLX_TTS_REF_AUDIO" | awk '{print $1}')" = \
        "b01ccb2c0ad5427f1478219b96dfba0d27e12bad281a81da46fed470a83630d4"
    test "$(sha256sum "$VH_AVATAR_REFERENCE" | awk '{print $1}')" = \
        "c8afa1d691711330e525db0048dd21e37ab71c46476a7504d0e75b4ae8fcd162"

    "$project_dir/.venv-gpu/bin/python" -c \
        "from virtual_human.config import Settings; assert Settings().offline_runtime"
    config_path="$("$project_dir/scripts/run-openviking-memory.sh" config)"
    if grep -Eq 'https?://(api\.|ark\.|[^\"/]*openai\.com)' "$config_path"; then
        echo "OpenViking config contains an external provider endpoint." >&2
        exit 1
    fi
}

start_models() {
    validate_start
    mkdir -p "$service_dir"
    cd "$project_dir"

    start_process \
        "embedding" "$embedding_pid" "$embedding_log" \
        "$project_dir/scripts/run-xgc-embedding.sh"
    wait_for_url "embedding" "http://127.0.0.1:${VH_EMBEDDING_PORT:-8002}/healthz" "$embedding_pid" 180

    start_process \
        "LLM" "$llm_pid" "$llm_log" \
        "$project_dir/scripts/run-xgc-llm.sh"
    wait_for_url "LLM" "http://127.0.0.1:${VH_LLM_PORT:-8000}/health" "$llm_pid" 600

    start_process \
        "ASR" "$asr_pid" "$asr_log" \
        "$project_dir/scripts/run-xgc-asr.sh"
    wait_for_url "ASR" "http://127.0.0.1:${VH_ASR_PORT:-8001}/healthz" "$asr_pid" 300

    start_process \
        "TTS" "$tts_pid" "$tts_log" \
        "$project_dir/scripts/run-xgc-tts.sh"
    wait_for_url "TTS" "http://127.0.0.1:${VH_TTS_PORT:-8010}/health" "$tts_pid" 600
}

start_app() {
    validate_start
    mkdir -p "$service_dir"
    cd "$project_dir"

    if [ "${VH_MEMORY_ENABLED:-false}" = "true" ]; then
        "$project_dir/scripts/run-openviking-memory.sh" start
    fi

    start_process \
        "FlashHead" "$renderer_pid" "$renderer_log" \
        "$project_dir/scripts/run-flashhead-gpu.sh"
    wait_for_url \
        "FlashHead" "http://127.0.0.1:${VH_FLASHHEAD_PORT}/healthz" "$renderer_pid" 600

    start_process \
        "gateway" "$gateway_pid" "$gateway_log" \
        "$project_dir/.venv-gpu/bin/virtual-human"
    wait_for_url "gateway" "http://127.0.0.1:${VH_PORT}/healthz" "$gateway_pid" 180
    echo "Public entry: use the Xiangongyun instance web URL."
}

start_services() {
    start_models
    start_app
}

stop_app() {
    stop_pid "gateway" "$gateway_pid"
    stop_pid "FlashHead" "$renderer_pid"
    if [ "${VH_MEMORY_ENABLED:-false}" = "true" ]; then
        "$project_dir/scripts/run-openviking-memory.sh" stop
    fi
}

stop_models() {
    stop_pid "TTS" "$tts_pid"
    stop_pid "ASR" "$asr_pid"
    stop_pid "LLM" "$llm_pid"
    stop_pid "embedding" "$embedding_pid"
}

stop_services() {
    stop_app
    stop_models
}

status_one() {
    name="$1"
    pid_file="$2"
    url="$3"
    if is_running "$pid_file" && curl -fsS "$url" >/dev/null 2>&1; then
        echo "$name: running pid=$(cat "$pid_file")"
        return 0
    fi
    echo "$name: stopped"
    return 1
}

status_services() {
    exit_code=0
    status_one "embedding" "$embedding_pid" \
        "http://127.0.0.1:${VH_EMBEDDING_PORT:-8002}/healthz" || exit_code=1
    status_one "LLM" "$llm_pid" \
        "http://127.0.0.1:${VH_LLM_PORT:-8000}/health" || exit_code=1
    status_one "ASR" "$asr_pid" \
        "http://127.0.0.1:${VH_ASR_PORT:-8001}/healthz" || exit_code=1
    status_one "TTS" "$tts_pid" \
        "http://127.0.0.1:${VH_TTS_PORT:-8010}/health" || exit_code=1
    if [ "${VH_MEMORY_ENABLED:-false}" = "true" ]; then
        "$project_dir/scripts/run-openviking-memory.sh" status || exit_code=1
    else
        echo "OpenViking: disabled"
    fi
    status_one "FlashHead" "$renderer_pid" \
        "http://127.0.0.1:${VH_FLASHHEAD_PORT}/healthz" || exit_code=1
    status_one "gateway" "$gateway_pid" \
        "http://127.0.0.1:${VH_PORT}/healthz" || exit_code=1
    return "$exit_code"
}

load_environment
case "${1:-status}" in
    start) start_services ;;
    models-start) start_models ;;
    app-start) start_app ;;
    stop) stop_services ;;
    models-stop) stop_models ;;
    app-stop) stop_app ;;
    restart)
        stop_services || true
        start_services
        ;;
    status) status_services ;;
    logs)
        echo "embedding: $embedding_log"
        echo "LLM: $llm_log"
        echo "ASR: $asr_log"
        echo "TTS: $tts_log"
        echo "OpenViking: $project_dir/runtime/openviking-girlfriend/server.log"
        echo "FlashHead: $renderer_log"
        echo "gateway: $gateway_log"
        ;;
    *)
        echo "usage: $0 {start|stop|restart|status|models-start|models-stop|app-start|app-stop|logs}" >&2
        exit 2
        ;;
esac
