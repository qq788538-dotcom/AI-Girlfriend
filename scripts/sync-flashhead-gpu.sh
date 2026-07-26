#!/bin/sh
set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 SSH_HOST SSH_PORT [SSH_USER]" >&2
    exit 2
fi

SSH_HOST="$1"
SSH_PORT="$2"
SSH_USER="${3:-root}"
PROJECT_DIR="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="${VH_GPU_PROJECT_DIR:-/root/virtual-human}"
SSH_IDENTITY="${VH_GPU_SSH_KEY:-$PROJECT_DIR/runtime/credentials/codex-openssh}"

if [ ! -f "$SSH_IDENTITY" ]; then
    echo "SSH identity not found: $SSH_IDENTITY" >&2
    exit 1
fi

ssh -i "$SSH_IDENTITY" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" \
    "mkdir -p '$REMOTE_DIR/src' '$REMOTE_DIR/scripts' '$REMOTE_DIR/vendor/SoulX-FlashHead'"

rsync -az \
    -e "ssh -i $SSH_IDENTITY -p $SSH_PORT" \
    "$PROJECT_DIR/pyproject.toml" \
    "$PROJECT_DIR/README.md" \
    "$SSH_USER@$SSH_HOST:$REMOTE_DIR/"

rsync -az \
    --exclude '__pycache__/' \
    -e "ssh -i $SSH_IDENTITY -p $SSH_PORT" \
    "$PROJECT_DIR/src/" \
    "$SSH_USER@$SSH_HOST:$REMOTE_DIR/src/"

rsync -az \
    -e "ssh -i $SSH_IDENTITY -p $SSH_PORT" \
    "$PROJECT_DIR/scripts/" \
    "$SSH_USER@$SSH_HOST:$REMOTE_DIR/scripts/"

rsync -az \
    --exclude '.git/' \
    --exclude '/models/' \
    --exclude '/examples/' \
    --exclude '/assets/' \
    --exclude '__pycache__/' \
    -e "ssh -i $SSH_IDENTITY -p $SSH_PORT" \
    "$PROJECT_DIR/vendor/SoulX-FlashHead/" \
    "$SSH_USER@$SSH_HOST:$REMOTE_DIR/vendor/SoulX-FlashHead/"

echo "Project source synchronized to $SSH_USER@$SSH_HOST:$REMOTE_DIR"
