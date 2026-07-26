#!/bin/sh
set -eu

: "${LIVEACT_CKPT_DIR:?Set LIVEACT_CKPT_DIR to the downloaded SoulX-LiveAct checkpoint directory}"
: "${LIVEACT_WAV2VEC_DIR:?Set LIVEACT_WAV2VEC_DIR to chinese-wav2vec2-base}"

LIVEACT_PORT="${LIVEACT_PORT:-5001}"
LIVEACT_SIZE="${LIVEACT_SIZE:-416*720}"
LIVEACT_VIDEO_DIR="${LIVEACT_VIDEO_DIR:-./generated_videos}"
LIVEACT_CUDA_DEVICES="${LIVEACT_CUDA_DEVICES:-0,1}"
LIVEACT_NPROC="${LIVEACT_NPROC:-2}"
LIVEACT_MASTER_PORT="${LIVEACT_MASTER_PORT:-29531}"
LIVEACT_FP8_KV_CACHE="${LIVEACT_FP8_KV_CACHE:-0}"
LIVEACT_BLOCK_OFFLOAD="${LIVEACT_BLOCK_OFFLOAD:-0}"
LIVEACT_T5_CPU="${LIVEACT_T5_CPU:-0}"

cd "$(dirname "$0")/../vendor/SoulX-LiveAct"

set -- torchrun \
  --nproc_per_node="$LIVEACT_NPROC" \
  --master_port="$LIVEACT_MASTER_PORT" \
  demo.py \
  --ckpt_dir "$LIVEACT_CKPT_DIR" \
  --wav2vec_dir "$LIVEACT_WAV2VEC_DIR" \
  --size "$LIVEACT_SIZE" \
  --port "$LIVEACT_PORT" \
  --video_save_path "$LIVEACT_VIDEO_DIR"

if [ "$LIVEACT_FP8_KV_CACHE" = "1" ]; then
  set -- "$@" --fp8_kv_cache
fi
if [ "$LIVEACT_BLOCK_OFFLOAD" = "1" ]; then
  set -- "$@" --block_offload
fi
if [ "$LIVEACT_T5_CPU" = "1" ]; then
  set -- "$@" --t5_cpu
fi

exec env USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES="$LIVEACT_CUDA_DEVICES" "$@"
