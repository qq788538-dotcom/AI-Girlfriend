#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
PROJECT_DIR="${VH_AUTODL_PROJECT_DIR:-/root/AI-Girlfriend}"
DATA_ROOT="${VH_AUTODL_DATA_ROOT:-/root/autodl-tmp}"
RUNTIME_ENV="${VH_AUTODL_RUNTIME_ENV:-$SCRIPT_DIR/runtime.env}"
SECRETS_ENV="${VH_AUTODL_SECRETS_ENV:-$SCRIPT_DIR/secrets.env}"
SERVICE_DIR="$DATA_ROOT/cloud-runtime"

declare -A PID_FILES=(
    [embedding]="$SERVICE_DIR/embedding.pid"
    [memory_llm]="$SERVICE_DIR/memory-llm.pid"
    [asr]="$SERVICE_DIR/asr.pid"
    [tts]="$SERVICE_DIR/tts.pid"
    [gateway]="$SERVICE_DIR/gateway.pid"
)
declare -A LOG_FILES=(
    [embedding]="$SERVICE_DIR/embedding.log"
    [memory_llm]="$SERVICE_DIR/memory-llm.log"
    [asr]="$SERVICE_DIR/asr.log"
    [tts]="$SERVICE_DIR/tts.log"
    [gateway]="$SERVICE_DIR/gateway.log"
)

load_environment() {
    if [ ! -f "$RUNTIME_ENV" ] || [ ! -f "$SECRETS_ENV" ]; then
        echo "Missing runtime.env or secrets.env in $SCRIPT_DIR" >&2
        exit 1
    fi
    set -a
    # shellcheck disable=SC1090
    . "$RUNTIME_ENV"
    # shellcheck disable=SC1090
    . "$SECRETS_ENV"
    set +a

    export VH_XGC_PROJECT_DIR="$PROJECT_DIR"
    export VH_GPU_PROJECT_DIR="$PROJECT_DIR"
    export VH_HOST="${VH_HOST:-0.0.0.0}"
    export VH_PORT="${VH_PORT:-6006}"
    export VH_PUBLIC_DIR="${VH_PUBLIC_DIR:-$PROJECT_DIR/public}"
    export VH_AVATAR_RENDERER_TOKEN_FILE="$PROJECT_DIR/runtime/credentials/avatar-renderer.token"
    export OPENVIKING_SERVER_BIN="${OPENVIKING_SERVER_BIN:-$PROJECT_DIR/.venv-openviking/bin/openviking-server}"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    unset OPENAI_API_KEY VH_ARK_API_KEY VH_MEMORY_EMBEDDING_API_KEY
}

is_running() {
    local pid_file="$1"
    local pid
    test -s "$pid_file" || return 1
    pid="$(cat "$pid_file")"
    kill -0 "$pid" 2>/dev/null
}

start_process() {
    local name="$1"
    local pid_file="$2"
    local log_file="$3"
    shift 3
    if is_running "$pid_file"; then
        echo "$name: already running (pid $(cat "$pid_file"))"
        return
    fi
    rm -f "$pid_file"
    nohup "$@" >"$log_file" 2>&1 </dev/null &
    echo "$!" >"$pid_file"
    echo "$name: starting (pid $(cat "$pid_file"))"
}

stop_process() {
    local name="$1"
    local pid_file="$2"
    local pid
    if ! is_running "$pid_file"; then
        rm -f "$pid_file"
        echo "$name: stopped"
        return
    fi
    pid="$(cat "$pid_file")"
    kill "$pid"
    for _ in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "$name did not stop cleanly (pid $pid)" >&2
        return 1
    fi
    rm -f "$pid_file"
    echo "$name: stopped"
}

wait_for_url() {
    local name="$1"
    local url="$2"
    local pid_file="${3:-}"
    local max_attempts="${4:-300}"
    local attempt
    for attempt in $(seq 1 "$max_attempts"); do
        if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
            echo "$name: ready"
            return
        fi
        if [ -n "$pid_file" ] && ! is_running "$pid_file"; then
            echo "$name exited before becoming ready; inspect its log" >&2
            exit 1
        fi
        sleep 1
    done
    echo "$name did not become ready: $url" >&2
    exit 1
}

require_loopback() {
    local name="$1"
    local endpoint="$2"
    case "$endpoint" in
        http://127.0.0.1/*|http://127.0.0.1:*|http://localhost/*|http://localhost:*|\
        ws://127.0.0.1/*|ws://127.0.0.1:*|ws://localhost/*|ws://localhost:*) ;;
        *)
            echo "$name must remain loopback-only: $endpoint" >&2
            exit 1
            ;;
    esac
}

validate_start() {
    if [ "${VH_OFFLINE_RUNTIME:-false}" != "true" ]; then
        echo "PRO 6000 cloud mode requires VH_OFFLINE_RUNTIME=true" >&2
        exit 1
    fi
    if [ "${VH_UPSTREAM_MODE:-}" != "omlx" ] || [ "${VH_CHAT_BACKEND:-}" != "omlx" ]; then
        echo "PRO 6000 cloud mode requires local omlx-compatible chat" >&2
        exit 1
    fi
    if [ "${VH_MEMORY_LOCAL_ONLY:-false}" != "true" ]; then
        echo "OpenViking private memory extraction must remain local" >&2
        exit 1
    fi

    require_loopback "LLM" "${VH_OMLX_BASE_URL:-}"
    require_loopback "ASR" "${VH_ASR_BASE_URL:-}"
    require_loopback "TTS" "${VH_TTS_BASE_URL:-}"
    require_loopback "Embedding" "${VH_MEMORY_EMBEDDING_BASE_URL:-}"
    require_loopback "OpenViking" "${VH_MEMORY_BASE_URL:-}"
    require_loopback "OpenViking VLM" "${VH_MEMORY_VLM_BASE_URL:-}"
    require_loopback "LiveAct renderer" "${VH_AVATAR_RENDERER_WS:-}"
    require_loopback "LiveAct media" "${VH_AVATAR_MEDIA_BASE_URL:-}"

    for executable in \
        "$PROJECT_DIR/.venv-gpu/bin/virtual-human" \
        "$PROJECT_DIR/.venv-asr/bin/virtual-human-asr" \
        "$PROJECT_DIR/.venv-vllm-omni/bin/vllm" \
        "$PROJECT_DIR/.venv-embedding/bin/virtual-human-embedding" \
        "$OPENVIKING_SERVER_BIN"; do
        test -x "$executable" || {
            echo "Missing runtime executable: $executable" >&2
            echo "Run deploy/autodl/bootstrap.sh first" >&2
            exit 1
        }
    done

    test "$(sha256sum "$VH_OMLX_TTS_REF_AUDIO" | awk '{print $1}')" = \
        "b01ccb2c0ad5427f1478219b96dfba0d27e12bad281a81da46fed470a83630d4"
    test "$(sha256sum "$VH_AVATAR_REFERENCE" | awk '{print $1}')" = \
        "c8afa1d691711330e525db0048dd21e37ab71c46476a7504d0e75b4ae8fcd162"
    "$PROJECT_DIR/.venv-gpu/bin/python" -c \
        "from virtual_human.config import Settings; s=Settings(); assert s.offline_runtime and s.chat_backend == 'omlx'"
}

start_liveact() {
    "$SCRIPT_DIR/liveact-control.sh" start
    wait_for_url "LiveAct renderer" "http://127.0.0.1:8772/healthz" "" 1200
}

start_models() {
    validate_start
    mkdir -p "$SERVICE_DIR"
    cd "$PROJECT_DIR"

    start_process embedding "${PID_FILES[embedding]}" "${LOG_FILES[embedding]}" \
        "$PROJECT_DIR/scripts/run-xgc-embedding.sh"
    wait_for_url embedding "http://127.0.0.1:${VH_EMBEDDING_PORT:-8002}/healthz" \
        "${PID_FILES[embedding]}" 300

    start_process memory-llm "${PID_FILES[memory_llm]}" "${LOG_FILES[memory_llm]}" \
        env \
        VH_LLM_VENV_DIR="$PROJECT_DIR/.venv-vllm-omni" \
        VH_LLM_MODEL_PATH="$PROJECT_DIR/runtime/models/Qwen3-4B-AWQ" \
        VH_OMLX_CHAT_MODEL="${VH_MEMORY_VLM_MODEL:-Qwen3-4B-AWQ}" \
        VH_LLM_GPU_MEMORY_UTILIZATION="${VH_MEMORY_LLM_GPU_MEMORY_UTILIZATION:-0.10}" \
        VH_LLM_MAX_MODEL_LEN="${VH_MEMORY_LLM_MAX_MODEL_LEN:-4096}" \
        VH_LLM_MAX_NUM_BATCHED_TOKENS="${VH_MEMORY_LLM_MAX_NUM_BATCHED_TOKENS:-4096}" \
        VH_LLM_ATTENTION_BACKEND="${VH_MEMORY_LLM_ATTENTION_BACKEND:-FLASH_ATTN}" \
        VH_LLM_KV_CACHE_DTYPE=auto \
        VH_LLM_QUANTIZATION=awq \
        VH_LLM_DTYPE=float16 \
        VH_LLM_TOOL_CALL_PARSER=qwen3_coder \
        "$PROJECT_DIR/scripts/run-xgc-llm.sh"
    wait_for_url memory-llm "http://127.0.0.1:${VH_LLM_PORT:-8000}/health" \
        "${PID_FILES[memory_llm]}" 900

    start_process ASR "${PID_FILES[asr]}" "${LOG_FILES[asr]}" \
        "$PROJECT_DIR/scripts/run-xgc-asr.sh"
    wait_for_url ASR "http://127.0.0.1:${VH_ASR_PORT:-8001}/healthz" \
        "${PID_FILES[asr]}" 600

    start_process TTS "${PID_FILES[tts]}" "${LOG_FILES[tts]}" \
        "$PROJECT_DIR/scripts/run-xgc-tts.sh"
    wait_for_url TTS "http://127.0.0.1:${VH_TTS_PORT:-8010}/health" \
        "${PID_FILES[tts]}" 900
}

start_app() {
    validate_start
    mkdir -p "$SERVICE_DIR"
    cd "$PROJECT_DIR"
    if [ "${VH_MEMORY_ENABLED:-false}" = true ]; then
        "$PROJECT_DIR/scripts/run-openviking-memory.sh" start
    fi
    start_process gateway "${PID_FILES[gateway]}" "${LOG_FILES[gateway]}" \
        "$PROJECT_DIR/.venv-gpu/bin/virtual-human"
    wait_for_url gateway "http://127.0.0.1:${VH_PORT:-6006}/healthz" \
        "${PID_FILES[gateway]}" 300
    echo "Public entry: use the AutoDL port 6006 custom-service HTTPS URL"
}

start_all() {
    start_liveact
    start_models
    start_app
}

stop_all() {
    stop_process gateway "${PID_FILES[gateway]}" || true
    if [ "${VH_MEMORY_ENABLED:-false}" = true ]; then
        "$PROJECT_DIR/scripts/run-openviking-memory.sh" stop || true
    fi
    stop_process TTS "${PID_FILES[tts]}" || true
    stop_process ASR "${PID_FILES[asr]}" || true
    stop_process memory-llm "${PID_FILES[memory_llm]}" || true
    stop_process embedding "${PID_FILES[embedding]}" || true
    "$SCRIPT_DIR/liveact-control.sh" stop || true
}

status_one() {
    local name="$1"
    local pid_file="$2"
    local url="$3"
    if is_running "$pid_file" && curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then
        echo "$name: running pid=$(cat "$pid_file")"
        return 0
    fi
    echo "$name: stopped"
    return 1
}

status_all() {
    local result=0
    curl -fsS --max-time 3 http://127.0.0.1:8772/healthz >/dev/null 2>&1 \
        && echo "LiveAct renderer: running" || { echo "LiveAct renderer: stopped"; result=1; }
    status_one embedding "${PID_FILES[embedding]}" \
        "http://127.0.0.1:${VH_EMBEDDING_PORT:-8002}/healthz" || result=1
    status_one memory-llm "${PID_FILES[memory_llm]}" \
        "http://127.0.0.1:${VH_LLM_PORT:-8000}/health" || result=1
    status_one ASR "${PID_FILES[asr]}" \
        "http://127.0.0.1:${VH_ASR_PORT:-8001}/healthz" || result=1
    status_one TTS "${PID_FILES[tts]}" \
        "http://127.0.0.1:${VH_TTS_PORT:-8010}/health" || result=1
    if [ "${VH_MEMORY_ENABLED:-false}" = true ]; then
        "$PROJECT_DIR/scripts/run-openviking-memory.sh" status || result=1
    fi
    status_one gateway "${PID_FILES[gateway]}" \
        "http://127.0.0.1:${VH_PORT:-6006}/healthz" || result=1
    return "$result"
}

load_environment
mkdir -p "$SERVICE_DIR"
case "${1:-status}" in
    start) start_all ;;
    stop) stop_all ;;
    restart) stop_all; start_all ;;
    models-start) start_liveact; start_models ;;
    app-start) start_app ;;
    status) status_all ;;
    logs)
        printf '%s\n' \
            "LiveAct: $DATA_ROOT/liveact-runtime/demo.log" \
            "embedding: ${LOG_FILES[embedding]}" \
            "memory-llm: ${LOG_FILES[memory_llm]}" \
            "ASR: ${LOG_FILES[asr]}" \
            "TTS: ${LOG_FILES[tts]}" \
            "OpenViking: $PROJECT_DIR/runtime/openviking-girlfriend/server.log" \
            "gateway: ${LOG_FILES[gateway]}"
        ;;
    *) echo "usage: $0 {start|stop|restart|status|models-start|app-start|logs}" >&2; exit 2 ;;
esac
