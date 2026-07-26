#!/bin/sh
set -eu

: "${LIVEACT_INPUT_JSON:?Set LIVEACT_INPUT_JSON to one locked offline input}"
: "${LIVEACT_OUTPUT_DIR:?Set LIVEACT_OUTPUT_DIR to the raw/final MP4 directory}"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RUNNER="${LIVEACT_RUNNER:-$SCRIPT_DIR/run-liveact-candidate.sh}"
FINALIZER="${LIVEACT_FINALIZER:-$SCRIPT_DIR/finalize-liveact-offline.sh}"
MANIFEST_PYTHON="${LIVEACT_MANIFEST_PYTHON:-python3}"
LOCKED_AVATAR_NAME="${LIVEACT_LOCKED_AVATAR_NAME:-avatar-ai-girlfriend-v6.png}"
LOCKED_AVATAR_SHA256="${LIVEACT_LOCKED_AVATAR_SHA256:-c8afa1d691711330e525db0048dd21e37ab71c46476a7504d0e75b4ae8fcd162}"

if [ ! -x "$RUNNER" ]; then
  echo "LiveAct runner is not executable: $RUNNER" >&2
  exit 2
fi
if [ ! -x "$FINALIZER" ]; then
  echo "LiveAct finalizer is not executable: $FINALIZER" >&2
  exit 2
fi

RAW_OUTPUT=$(
  "$MANIFEST_PYTHON" - \
    "$LIVEACT_INPUT_JSON" \
    "$LIVEACT_OUTPUT_DIR" \
    "$LOCKED_AVATAR_NAME" \
    "$LOCKED_AVATAR_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).expanduser()
output_dir = Path(sys.argv[2]).expanduser()
locked_name = sys.argv[3]
locked_digest = sys.argv[4]

payload = json.loads(manifest_path.read_text(encoding="utf-8"))
if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
    raise SystemExit("Offline manifest must contain exactly one input object")
entry = payload[0]
image = Path(str(entry.get("cond_image", ""))).expanduser()
audio = Path(str(entry.get("cond_audio", ""))).expanduser()
if image.name != locked_name or not image.is_file():
    raise SystemExit("LiveAct refused a missing or unlocked avatar image")
digest = hashlib.sha256(image.read_bytes()).hexdigest()
if digest != locked_digest:
    raise SystemExit("LiveAct refused an avatar whose identity digest is not locked")
if not audio.is_file():
    raise SystemExit("LiveAct input audio does not exist")

image_stem = image.name.split(".", 1)[0]
audio_stem = audio.name.split(".", 1)[0]
print(output_dir / f"{image_stem}_{audio_stem}.mp4")
PY
)

"$RUNNER"

if [ ! -f "$RAW_OUTPUT" ]; then
  echo "LiveAct finished without the expected raw output: $RAW_OUTPUT" >&2
  exit 1
fi

FINAL_OUTPUT="${LIVEACT_FINAL_OUTPUT:-${RAW_OUTPUT%.mp4}.final.mp4}"
"$FINALIZER" "$RAW_OUTPUT" "$FINAL_OUTPUT"
if [ ! -f "$FINAL_OUTPUT" ]; then
  echo "LiveAct finalizer did not create the expected output: $FINAL_OUTPUT" >&2
  exit 1
fi
echo "Verified offline LiveAct result: $FINAL_OUTPUT"
