#!/bin/sh
set -eu

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON_BIN="${VH_HIGGS_PYTHON:-$PROJECT_ROOT/runtime/moss-mlx/.venv/bin/python}"
MODEL="$PROJECT_ROOT/runtime/higgs-mlx/models/Higgs-TTS-3-4B-bf16"
REFERENCE_AUDIO="$PROJECT_ROOT/runtime/voice-calibration/reference-female-only-complete-11s.wav"
REFERENCE_TEXT="要是你跟别人聊天，我会吃醋的哦。我都开始胡思乱想了。我不知道什么是皮老板，只是你不在的时候，我就一直等你。"
HOST="${VH_HIGGS_HOST:-127.0.0.1}"
PORT="${VH_HIGGS_PORT:-8010}"

exec env PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" -m virtual_human.higgs_server \
    --model "$MODEL" \
    --reference-audio "$REFERENCE_AUDIO" \
    --reference-text "$REFERENCE_TEXT" \
    --host "$HOST" \
    --port "$PORT" \
    --seed 20260817 \
    --temperature 0.8 \
    --top-p 0.95 \
    --top-k 50
