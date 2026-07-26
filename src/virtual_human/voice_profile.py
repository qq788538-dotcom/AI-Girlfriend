from __future__ import annotations

import hashlib
from pathlib import Path

# This is the user-approved production voice. Changing any constant is a voice
# migration and requires the user's explicit approval.
LOCKED_TTS_BASE_URL = "http://127.0.0.1:8010/v1"
LOCKED_TTS_MODEL = "Higgs-TTS-3-4B-bf16"
LOCKED_TTS_TEMPERATURE = 0.8
LOCKED_TTS_TOP_P = 0.95
LOCKED_TTS_FLOW_MODE = "g_continuous"
LOCKED_TTS_SEED = 20260817
LOCKED_TTS_TOP_K = 50

LOCKED_VOICE_REFERENCE = Path(
    "runtime/voice-calibration/reference-female-only-complete-11s.wav"
)
LOCKED_VOICE_REFERENCE_SHA256 = (
    "b01ccb2c0ad5427f1478219b96dfba0d27e12bad281a81da46fed470a83630d4"
)
LOCKED_VOICE_REFERENCE_TEXT = (
    "要是你跟别人聊天，我会吃醋的哦。我都开始胡思乱想了。"
    "我不知道什么是皮老板，只是你不在的时候，我就一直等你。"
)


def verify_locked_voice_reference(path: Path = LOCKED_VOICE_REFERENCE) -> str:
    if not path.is_file():
        raise ValueError(f"Locked voice reference is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != LOCKED_VOICE_REFERENCE_SHA256:
        raise ValueError(
            "Voice profile is locked. Refusing a changed or noisy reference audio."
        )
    return digest
