#!/bin/sh
set -eu

OMLX_BIN="${VH_OMLX_BIN:-/opt/homebrew/bin/omlx}"
MODEL_DIR="${VH_OMLX_MODEL_DIR:-$HOME/.omlx/models}"
HOST="${VH_OMLX_HOST:-127.0.0.1}"
PORT="${VH_OMLX_PORT:-8000}"
MAX_CONCURRENT_REQUESTS="${VH_OMLX_MAX_CONCURRENT_REQUESTS:-1}"
MEMORY_GUARD="${VH_OMLX_MEMORY_GUARD:-balanced}"
HOT_CACHE_SIZE="${VH_OMLX_HOT_CACHE_SIZE:-4GB}"
SSD_CACHE_SIZE="${VH_OMLX_SSD_CACHE_SIZE:-20GB}"

exec "$OMLX_BIN" serve \
    --model-dir "$MODEL_DIR" \
    --host "$HOST" \
    --port "$PORT" \
    --max-concurrent-requests "$MAX_CONCURRENT_REQUESTS" \
    --memory-guard "$MEMORY_GUARD" \
    --hot-cache-max-size "$HOT_CACHE_SIZE" \
    --paged-ssd-cache-max-size "$SSD_CACHE_SIZE"
