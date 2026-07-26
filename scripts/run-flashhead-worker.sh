#!/bin/sh
set -eu

: "${VH_FLASHHEAD_CKPT_DIR:?Set VH_FLASHHEAD_CKPT_DIR to SoulX-FlashHead-1_3B}"
: "${VH_FLASHHEAD_WAV2VEC_DIR:?Set VH_FLASHHEAD_WAV2VEC_DIR to wav2vec2-base-960h}"

PROJECT_DIR="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_DIR"
exec python -m virtual_human.flashhead_worker
