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
    if [ "${VH_MEMORY_LOCAL_ONLY:-false}" = "true" ]; then
        embedding_base="${VH_MEMORY_EMBEDDING_BASE_URL:-http://127.0.0.1:8002/v1}"
        embedding_model="${VH_MEMORY_EMBEDDING_MODEL:-Qwen3-Embedding-0.6B}"
        embedding_dimension="${VH_MEMORY_EMBEDDING_DIMENSION:-1024}"
        llm_base="${VH_MEMORY_VLM_BASE_URL:-http://127.0.0.1:8000/v1}"
        llm_model="${VH_MEMORY_VLM_MODEL:-Qwen3.6-35B-A3B-AWQ}"
        llm_max_tokens="${VH_MEMORY_VLM_MAX_TOKENS:-1024}"
        llm_max_concurrent="${VH_MEMORY_VLM_MAX_CONCURRENT:-1}"
        case "$embedding_base" in
            http://127.0.0.1/*|http://127.0.0.1:*|http://localhost/*|http://localhost:*) ;;
            *)
                echo "Local-only memory requires a loopback embedding endpoint" >&2
                exit 1
                ;;
        esac
        case "$llm_base" in
            http://127.0.0.1/*|http://127.0.0.1:*|http://localhost/*|http://localhost:*) ;;
            *)
                echo "Local-only memory requires a loopback VLM endpoint" >&2
                exit 1
                ;;
        esac
        jq -n \
            --arg workspace "$workspace_dir" \
            --arg embedding_base "$embedding_base" \
            --arg embedding_model "$embedding_model" \
            --argjson embedding_dimension "$embedding_dimension" \
            --arg llm_base "$llm_base" \
            --arg llm_model "$llm_model" \
            --argjson llm_max_tokens "$llm_max_tokens" \
            --argjson llm_max_concurrent "$llm_max_concurrent" \
            '{
                storage: {
                    workspace: $workspace,
                    vectordb: {backend: "local"},
                    agfs: {backend: "local"}
                },
                embedding: {
                    max_concurrent: 4,
                    max_retries: 1,
                    dense: {
                        provider: "openai",
                        api_base: $embedding_base,
                        api_key: "local-offline",
                        model: $embedding_model,
                        dimension: $embedding_dimension,
                        input: "text",
                        encoding_format: "float"
                    }
                },
                vlm: {
                    provider: "openai",
                    api_base: $llm_base,
                    api_key: "local-offline",
                    model: $llm_model,
                    thinking: false,
                    max_tokens: $llm_max_tokens,
                    max_concurrent: $llm_max_concurrent,
                    max_retries: 1
                },
                memory: {
                    version: "v3",
                    eager_prefetch: true,
                    session_skill_extraction_enabled: false,
                    link_enabled: false
                },
                server: {auth_mode: "dev"}
            }' > "$config_file"
        chmod 600 "$config_file"
        return
    fi

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
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
    TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" \
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
