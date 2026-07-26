#!/bin/sh
set -eu

: "${LIVEACT_CKPT_DIR:?Set LIVEACT_CKPT_DIR to the SoulX-LiveAct checkpoint directory}"
: "${LIVEACT_WAV2VEC_DIR:?Set LIVEACT_WAV2VEC_DIR to chinese-wav2vec2-base}"
: "${LIVEACT_INPUT_JSON:?Set LIVEACT_INPUT_JSON to the locked A/B input JSON}"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
LIVEACT_ROOT="${LIVEACT_ROOT:-$PROJECT_DIR/vendor/SoulX-LiveAct}"
LIVEACT_PYTHON="${LIVEACT_PYTHON:-python}"
LIVEACT_OUTPUT_DIR="${LIVEACT_OUTPUT_DIR:-$PROJECT_DIR/runtime/liveact-candidate}"
# The locked avatar is square. 512x512 preserves its framing and uses fewer
# latent tokens than the official 416x720 portrait consumer-GPU example.
LIVEACT_SIZE="${LIVEACT_SIZE:-512*512}"
LIVEACT_FPS="${LIVEACT_FPS:-24}"
LIVEACT_SEED="${LIVEACT_SEED:-42}"
LIVEACT_AUDIO_CFG="${LIVEACT_AUDIO_CFG:-1.0}"
LIVEACT_CUDA_DEVICES="${LIVEACT_CUDA_DEVICES:-0}"
LIVEACT_NPROC="${LIVEACT_NPROC:-1}"
LIVEACT_MASTER_PORT="${LIVEACT_MASTER_PORT:-29541}"
# On the current 48 GB RTX 4090 D node, FP8 KV becomes corrupted exactly when
# ConvKV first updates the recurrent cache (the third generated block). Offline
# quality is the target, so keep the cache in BF16 and offload it to CPU.
LIVEACT_FP8_KV_CACHE="${LIVEACT_FP8_KV_CACHE:-0}"
LIVEACT_BLOCK_OFFLOAD="${LIVEACT_BLOCK_OFFLOAD:-1}"
LIVEACT_T5_CPU="${LIVEACT_T5_CPU:-1}"
LIVEACT_OFFLOAD_CACHE="${LIVEACT_OFFLOAD_CACHE:-1}"
LIVEACT_STREAM_AUDIO="${LIVEACT_STREAM_AUDIO:-0}"
LIVEACT_MEAN_MEMORY="${LIVEACT_MEAN_MEMORY:-0}"
LIVEACT_DURA_PRINT="${LIVEACT_DURA_PRINT:-1}"

case "$LIVEACT_ROOT" in
  /*) ;;
  *) LIVEACT_ROOT="$PROJECT_DIR/$LIVEACT_ROOT" ;;
esac
case "$LIVEACT_CKPT_DIR" in
  /*) ;;
  *) LIVEACT_CKPT_DIR="$PROJECT_DIR/$LIVEACT_CKPT_DIR" ;;
esac
case "$LIVEACT_WAV2VEC_DIR" in
  /*) ;;
  *) LIVEACT_WAV2VEC_DIR="$PROJECT_DIR/$LIVEACT_WAV2VEC_DIR" ;;
esac
case "$LIVEACT_INPUT_JSON" in
  /*) ;;
  *) LIVEACT_INPUT_JSON="$PROJECT_DIR/$LIVEACT_INPUT_JSON" ;;
esac
case "$LIVEACT_OUTPUT_DIR" in
  /*) ;;
  *) LIVEACT_OUTPUT_DIR="$PROJECT_DIR/$LIVEACT_OUTPUT_DIR" ;;
esac

if [ ! -f "$LIVEACT_ROOT/generate.py" ]; then
  echo "SoulX-LiveAct generate.py not found: $LIVEACT_ROOT" >&2
  exit 2
fi
if [ ! -d "$LIVEACT_CKPT_DIR" ]; then
  echo "SoulX-LiveAct checkpoint directory not found: $LIVEACT_CKPT_DIR" >&2
  exit 2
fi
if [ ! -d "$LIVEACT_WAV2VEC_DIR" ]; then
  echo "Wav2Vec checkpoint directory not found: $LIVEACT_WAV2VEC_DIR" >&2
  exit 2
fi
if [ ! -f "$LIVEACT_INPUT_JSON" ]; then
  echo "LiveAct input JSON not found: $LIVEACT_INPUT_JSON" >&2
  exit 2
fi

mkdir -p "$LIVEACT_OUTPUT_DIR"
cd "$LIVEACT_OUTPUT_DIR"

set -- \
  "$LIVEACT_ROOT/generate.py" \
  --size "$LIVEACT_SIZE" \
  --ckpt_dir "$LIVEACT_CKPT_DIR" \
  --wav2vec_dir "$LIVEACT_WAV2VEC_DIR" \
  --fps "$LIVEACT_FPS" \
  --audio_cfg "$LIVEACT_AUDIO_CFG" \
  --input_json "$LIVEACT_INPUT_JSON" \
  --seed "$LIVEACT_SEED"

if [ "$LIVEACT_FP8_KV_CACHE" = "1" ]; then
  set -- "$@" --fp8_kv_cache
fi
if [ "$LIVEACT_BLOCK_OFFLOAD" = "1" ]; then
  set -- "$@" --block_offload
fi
if [ "$LIVEACT_T5_CPU" = "1" ]; then
  set -- "$@" --t5_cpu
fi
if [ "$LIVEACT_OFFLOAD_CACHE" = "1" ]; then
  set -- "$@" --offload_cache
fi
if [ "$LIVEACT_STREAM_AUDIO" = "1" ]; then
  set -- "$@" --steam_audio
fi
if [ "$LIVEACT_MEAN_MEMORY" = "1" ]; then
  set -- "$@" --mean_memory
fi
if [ "$LIVEACT_DURA_PRINT" = "1" ]; then
  set -- "$@" --dura_print
fi

if [ "$LIVEACT_NPROC" -eq 1 ]; then
  env \
    USE_CHANNELS_LAST_3D=1 \
    CUDA_VISIBLE_DEVICES="$LIVEACT_CUDA_DEVICES" \
    "$LIVEACT_PYTHON" "$@"
  exit $?
fi

TORCHRUN_BIN=$(dirname "$LIVEACT_PYTHON")/torchrun
if [ ! -x "$TORCHRUN_BIN" ]; then
  TORCHRUN_BIN=torchrun
fi

env \
  USE_CHANNELS_LAST_3D=1 \
  CUDA_VISIBLE_DEVICES="$LIVEACT_CUDA_DEVICES" \
  "$TORCHRUN_BIN" \
  --nproc_per_node="$LIVEACT_NPROC" \
  --master_port="$LIVEACT_MASTER_PORT" \
  "$@"
