#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 SSH_HOST SSH_PORT" >&2
    exit 2
fi

SSH_HOST="$1"
SSH_PORT="$2"
LOCAL_PORT="${VH_FLASHHEAD_TUNNEL_LOCAL_PORT:-8771}"
REMOTE_PORT=8770

if command -v lsof >/dev/null 2>&1 &&
    lsof -nP -iTCP:"$LOCAL_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Local port $LOCAL_PORT is already in use; refusing to create an ambiguous tunnel." >&2
    echo "Inspect it with: lsof -nP -iTCP:$LOCAL_PORT -sTCP:LISTEN" >&2
    exit 3
fi

exec ssh \
    -N \
    -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" \
    -p "$SSH_PORT" \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=20 \
    -o ServerAliveCountMax=6 \
    -o StrictHostKeyChecking=accept-new \
    "root@$SSH_HOST"
