#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 RAW_LIVEACT_MP4 FINAL_MP4" >&2
  exit 2
fi

SOURCE=$1
OUTPUT=$2
LIVEACT_FFMPEG="${LIVEACT_FFMPEG:-ffmpeg}"
LIVEACT_FFPROBE="${LIVEACT_FFPROBE:-ffprobe}"
LIVEACT_AUDIO_ADVANCE_MS="${LIVEACT_AUDIO_ADVANCE_MS:-80}"
LIVEACT_VIDEO_SHARPEN="${LIVEACT_VIDEO_SHARPEN:-1}"
LIVEACT_VIDEO_CRF="${LIVEACT_VIDEO_CRF:-16}"
LIVEACT_VERIFY="${LIVEACT_VERIFY:-1}"

if [ ! -f "$SOURCE" ]; then
  echo "LiveAct source video not found: $SOURCE" >&2
  exit 2
fi
if [ "$SOURCE" = "$OUTPUT" ]; then
  echo "Source and output must be different paths" >&2
  exit 2
fi
case "$LIVEACT_AUDIO_ADVANCE_MS" in
  ''|*[!0-9]*) echo "LIVEACT_AUDIO_ADVANCE_MS must be a non-negative integer" >&2; exit 2 ;;
esac
case "$LIVEACT_VIDEO_SHARPEN" in
  0|1) ;;
  *) echo "LIVEACT_VIDEO_SHARPEN must be 0 or 1" >&2; exit 2 ;;
esac
case "$LIVEACT_VIDEO_CRF" in
  ''|*[!0-9]*) echo "LIVEACT_VIDEO_CRF must be a non-negative integer" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
AUDIO_ADVANCE_SECONDS=$(awk "BEGIN { printf \"%.6f\", $LIVEACT_AUDIO_ADVANCE_MS / 1000 }")
TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/liveact-finalize.XXXXXX")
trap 'rm -rf "$TEMP_DIR"' EXIT HUP INT TERM
VIDEO_TEMP="$TEMP_DIR/video.mp4"
AUDIO_TEMP="$TEMP_DIR/audio.m4a"
VIDEO_DURATION=$(
  "$LIVEACT_FFPROBE" \
    -v error \
    -select_streams v:0 \
    -show_entries stream=duration \
    -of default=noprint_wrappers=1:nokey=1 \
    "$SOURCE"
)
VIDEO_FRAMES=$(
  "$LIVEACT_FFPROBE" \
    -v error \
    -select_streams v:0 \
    -count_frames \
    -show_entries stream=nb_read_frames \
    -of default=noprint_wrappers=1:nokey=1 \
    "$SOURCE"
)
case "$VIDEO_DURATION" in
  ''|N/A) echo "Could not read source video duration" >&2; exit 2 ;;
esac
case "$VIDEO_FRAMES" in
  ''|N/A|*[!0-9]*) echo "Could not count source video frames" >&2; exit 2 ;;
esac

mkdir -p "$(dirname -- "$OUTPUT")"
set -- "$LIVEACT_FFMPEG" \
  -y \
  -hide_banner \
  -loglevel error \
  -i "$SOURCE" \
  -map 0:v:0 \
  -an

if [ "$LIVEACT_VIDEO_SHARPEN" = "1" ]; then
  set -- "$@" \
    -vf "unsharp=5:5:0.35:5:5:0" \
    -c:v libx264 \
    -preset slow \
    -crf "$LIVEACT_VIDEO_CRF" \
    -pix_fmt yuv420p
else
  set -- "$@" -c:v copy
fi

set -- "$@" \
  -frames:v "$VIDEO_FRAMES" \
  "$VIDEO_TEMP"
"$@"

"$LIVEACT_FFMPEG" \
  -y \
  -hide_banner \
  -loglevel error \
  -i "$SOURCE" \
  -vn \
  -af "atrim=start=$AUDIO_ADVANCE_SECONDS,asetpts=PTS-STARTPTS,apad=whole_dur=$VIDEO_DURATION,atrim=duration=$VIDEO_DURATION" \
  -c:a aac \
  -b:a 128k \
  -ar 24000 \
  -ac 1 \
  -t "$VIDEO_DURATION" \
  "$AUDIO_TEMP"

"$LIVEACT_FFMPEG" \
  -y \
  -hide_banner \
  -loglevel error \
  -i "$VIDEO_TEMP" \
  -i "$AUDIO_TEMP" \
  -map 0:v:0 \
  -map 1:a:0 \
  -c copy \
  -movflags +faststart \
  "$OUTPUT"

if [ "$LIVEACT_VERIFY" = "1" ]; then
  python3 "$PROJECT_DIR/scripts/benchmark-media-continuity.py" "$OUTPUT"
fi

echo "Final LiveAct video: $OUTPUT"
