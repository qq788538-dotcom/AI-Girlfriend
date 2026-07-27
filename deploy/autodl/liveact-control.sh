#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
SELF="$SCRIPT_DIR/$(basename "$0")"
PROJECT_DIR="${VH_AUTODL_PROJECT_DIR:-/root/AI-Girlfriend}"
DATA_ROOT="${VH_AUTODL_DATA_ROOT:-/root/autodl-tmp}"
LIVEACT_DIR="$PROJECT_DIR/vendor/SoulX-LiveAct"
LIVEACT_VENV="$PROJECT_DIR/.venv-liveact"
MODEL_DIR="$DATA_ROOT/models/LiveAct"
WAV2VEC_DIR="$DATA_ROOT/models/chinese-wav2vec2-base"
RUNTIME_DIR="$DATA_ROOT/liveact-runtime"
DEMO_PID_FILE="$RUNTIME_DIR/demo.pid"
WRAPPER_PID_FILE="$RUNTIME_DIR/wrapper.pid"
SUPERVISOR_PID_FILE="$RUNTIME_DIR/supervisor.pid"
DEMO_LOG="$RUNTIME_DIR/demo.log"
WRAPPER_LOG="$RUNTIME_DIR/wrapper.log"
SUPERVISOR_LOG="$RUNTIME_DIR/supervisor.log"
LIVEACT_PROMPT="${VH_LIVEACT_PROMPT:-A beautiful woman is speaking naturally, subtle expression, eye contact, realistic movement.}"

mkdir -p "$RUNTIME_DIR" "$DATA_ROOT/liveact-generated"

pid_is_alive() {
    local pid_file="$1"
    local pid
    test -s "$pid_file" || return 1
    pid="$(cat "$pid_file")"
    kill -0 "$pid" 2>/dev/null
}

stop_pid() {
    local pid_file="$1"
    local pid
    test -s "$pid_file" || return 0
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        for _ in $(seq 1 30); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid"
        fi
    fi
    rm -f "$pid_file"
}

require_installation() {
    test -x "$LIVEACT_VENV/bin/python"
    test -x "$PROJECT_DIR/.venv-gpu/bin/python"
    test -f "$MODEL_DIR/config.json"
    test -d "$WAV2VEC_DIR"
}

start_wrapper() {
    if pid_is_alive "$WRAPPER_PID_FILE"; then
        return 0
    fi
    cd "$PROJECT_DIR"
    nohup env \
        PYTHONPATH="$PROJECT_DIR/src" \
        VH_RENDERER_HOST=127.0.0.1 \
        VH_RENDERER_PORT=8772 \
        VH_RENDERER_BACKEND=liveact-official \
        VH_RENDERER_RUNTIME_DIR="$DATA_ROOT/liveact-renderer" \
        VH_LIVEACT_DEMO_URL=http://127.0.0.1:5001 \
        VH_LIVEACT_PROMPT="$LIVEACT_PROMPT" \
        VH_RENDERER_FPS=20 \
        "$PROJECT_DIR/.venv-gpu/bin/python" -m virtual_human.renderer_worker \
        >>"$WRAPPER_LOG" 2>&1 </dev/null &
    echo "$!" >"$WRAPPER_PID_FILE"
}

wait_and_start_wrapper() {
    for _ in $(seq 1 360); do
        if curl -fsS --max-time 2 http://127.0.0.1:5001/ >/dev/null 2>&1; then
            start_wrapper
            exit 0
        fi
        pid_is_alive "$DEMO_PID_FILE" || exit 1
        sleep 5
    done
    exit 1
}

start_services() {
    require_installation
    if ! pid_is_alive "$DEMO_PID_FILE"; then
        cd "$LIVEACT_DIR"
        nohup env \
            VH_AUTODL_PROJECT_DIR="$PROJECT_DIR" \
            VH_LIVEACT_FORCE_SDPA="${VH_LIVEACT_FORCE_SDPA:-1}" \
            VH_LIVEACT_CACHE_T5="${VH_LIVEACT_CACHE_T5:-1}" \
            VH_LIVEACT_T5_CACHE_ENTRIES="${VH_LIVEACT_T5_CACHE_ENTRIES:-32}" \
            VH_LIVEACT_CACHE_REFERENCE="${VH_LIVEACT_CACHE_REFERENCE:-1}" \
            VH_LIVEACT_REFERENCE_CACHE_ENTRIES="${VH_LIVEACT_REFERENCE_CACHE_ENTRIES:-4}" \
            VH_LIVEACT_WARMUP_REFERENCE="${VH_LIVEACT_WARMUP_REFERENCE:-$PROJECT_DIR/public/avatar-ai-girlfriend-v6.png}" \
            VH_LIVEACT_VAE_COMPILE_MODE="${VH_LIVEACT_VAE_COMPILE_MODE:-off}" \
            VH_LIVEACT_FIX_TAIL_FRAMES="${VH_LIVEACT_FIX_TAIL_FRAMES:-1}" \
            VH_LIVEACT_WARMUP_PROMPT="$LIVEACT_PROMPT" \
            PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
            USE_CHANNELS_LAST_3D=1 \
            CUDA_VISIBLE_DEVICES=0 \
            "$LIVEACT_VENV/bin/python" \
            "$PROJECT_DIR/deploy/autodl/liveact_demo_launcher.py" \
            --ckpt_dir "$MODEL_DIR" \
            --wav2vec_dir "$WAV2VEC_DIR" \
            --size "${VH_LIVEACT_SIZE:-416*720}" \
            --port 5001 \
            --fp8_kv_cache \
            --block_offload \
            --t5_cpu \
            --video_save_path "$DATA_ROOT/liveact-generated" \
            >>"$DEMO_LOG" 2>&1 </dev/null &
        echo "$!" >"$DEMO_PID_FILE"
    fi
    if curl -fsS --max-time 2 http://127.0.0.1:5001/ >/dev/null 2>&1; then
        start_wrapper
    elif ! pid_is_alive "$SUPERVISOR_PID_FILE"; then
        nohup "$SELF" wait-wrapper >>"$SUPERVISOR_LOG" 2>&1 </dev/null &
        echo "$!" >"$SUPERVISOR_PID_FILE"
    fi
}

show_status() {
    pid_is_alive "$DEMO_PID_FILE" && echo "liveact-demo: running" || echo "liveact-demo: stopped"
    pid_is_alive "$WRAPPER_PID_FILE" && echo "renderer-wrapper: running" || echo "renderer-wrapper: stopped"
    if curl -fsS --max-time 2 http://127.0.0.1:5001/ >/dev/null 2>&1; then
        echo "liveact-http: healthy"
    else
        echo "liveact-http: loading"
    fi
    if curl -fsS --max-time 2 http://127.0.0.1:8772/healthz; then
        echo
    else
        echo "renderer-http: unavailable"
    fi
}

case "${1:-status}" in
    start)
        start_services
        ;;
    wait-wrapper)
        wait_and_start_wrapper
        ;;
    stop)
        stop_pid "$SUPERVISOR_PID_FILE"
        stop_pid "$WRAPPER_PID_FILE"
        stop_pid "$DEMO_PID_FILE"
        ;;
    restart)
        "$SELF" stop
        "$SELF" start
        ;;
    status)
        show_status
        ;;
    logs)
        tail -n 80 "$DEMO_LOG" "$WRAPPER_LOG" "$SUPERVISOR_LOG" 2>/dev/null || true
        ;;
    *)
        echo "usage: $0 {start|stop|restart|status|logs}" >&2
        exit 2
        ;;
esac
