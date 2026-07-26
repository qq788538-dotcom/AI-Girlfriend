#!/bin/sh
set -eu

BASE_URL="${VH_OMLX_BASE_URL:-http://127.0.0.1:8000/v1}"
SETTINGS_PATH="${VH_OMLX_SETTINGS_PATH:-$HOME/.omlx/settings.json}"
CHAT_MODEL="${VH_OMLX_CHAT_MODEL:-Qwen3.5-9B-MLX-4bit}"
STT_MODEL="${VH_OMLX_STT_MODEL:-Qwen3-ASR-0.6B-4bit}"
TTS_MODEL="${VH_OMLX_TTS_MODEL:-Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit}"
TTS_VOICE="${VH_OMLX_TTS_VOICE:-Vivian}"

if [ ! -f "$SETTINGS_PATH" ]; then
    echo "oMLX settings not found: $SETTINGS_PATH" >&2
    exit 1
fi

API_KEY="$(jq -r '.auth.api_key // empty' "$SETTINGS_PATH")"
if [ -z "$API_KEY" ]; then
    echo "oMLX API key is missing from $SETTINGS_PATH" >&2
    exit 1
fi

WARM_DIR="$(mktemp -d)"
trap 'rm -rf "$WARM_DIR"' EXIT
WARM_WAV="$WARM_DIR/warm.wav"

attempt=0
until curl -fsS -H "Authorization: Bearer $API_KEY" "$BASE_URL/models" >/dev/null; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 60 ]; then
        echo "oMLX did not become ready at $BASE_URL" >&2
        exit 1
    fi
    sleep 1
done

echo "Warming local MLX TTS..."
jq -n \
    --arg model "$TTS_MODEL" \
    --arg voice "$TTS_VOICE" \
    '{
        model: $model,
        input: "你好，很高兴见到你。",
        voice: $voice,
        language: "Chinese",
        response_format: "wav",
        stream: true,
        streaming_interval: 0.5
    }' |
    curl -fsS \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        --data-binary @- \
        -o "$WARM_WAV" \
        "$BASE_URL/audio/speech"

echo "Warming local MLX ASR..."
curl -fsS \
    -H "Authorization: Bearer $API_KEY" \
    -F "file=@$WARM_WAV" \
    -F "model=$STT_MODEL" \
    -F "language=zh" \
    -F "response_format=json" \
    "$BASE_URL/audio/transcriptions" >/dev/null

echo "Warming local MLX chat..."
jq -n \
    --arg model "$CHAT_MODEL" \
    '{
        model: $model,
        messages: [
            {role: "system", content: "只简短回答。"},
            {role: "user", content: "你好"}
        ],
        chat_template_kwargs: {enable_thinking: false},
        max_tokens: 8
    }' |
    curl -fsS \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        --data-binary @- \
        "$BASE_URL/chat/completions" >/dev/null

echo "oMLX local models are warm: $STT_MODEL, $CHAT_MODEL, $TTS_MODEL"
