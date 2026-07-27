import pytest
from pydantic import ValidationError

from virtual_human.config import Settings
from virtual_human.voice_profile import (
    LOCKED_VOICE_REFERENCE,
    LOCKED_VOICE_REFERENCE_TEXT,
)


def test_client_persona_and_voice_are_environment_configurable() -> None:
    settings = Settings(
        VH_OPENAI_VOICE="coral",
        VH_VAD_EAGERNESS="high",
        VH_PERSONA_NAME="小岚",
        VH_PERSONA_INSTRUCTIONS="自然地聊天。",
    )

    assert settings.openai_voice == "coral"
    assert settings.vad_eagerness == "high"
    assert settings.persona_name == "小岚"
    assert settings.persona_instructions == "自然地聊天。"
    assert settings.tts_flow_mode == "g_continuous"
    assert settings.barge_in_enabled is False
    assert settings.resolved_tts_base_url == "http://127.0.0.1:8010/v1"
    assert settings.voice_profile_locked is True


def test_invalid_vad_eagerness_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(VH_VAD_EAGERNESS="extreme")


def test_tts_reference_audio_and_text_must_be_configured_together() -> None:
    with pytest.raises(ValidationError):
        Settings(
            VH_VOICE_PROFILE_LOCKED=False,
            VH_OMLX_TTS_REF_AUDIO="reference.wav",
            VH_OMLX_TTS_REF_TEXT="",
        )


def test_settings_reads_omlx_api_key_without_copying_it_to_env(tmp_path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text('{"auth":{"api_key":"local-only"}}')

    settings = Settings(
        VH_UPSTREAM_MODE="omlx",
        VH_OMLX_SETTINGS_PATH=settings_path,
    )

    assert settings.resolved_omlx_api_key() == "local-only"


def test_ark_chat_backend_requires_its_own_api_key() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            VH_CHAT_BACKEND="ark",
            VH_ARK_API_KEY="",
        )


def test_ark_chat_model_is_reported_as_the_resolved_chat_model() -> None:
    settings = Settings(
        _env_file=None,
        VH_CHAT_BACKEND="ark",
        VH_ARK_API_KEY="cloud-secret",
        VH_ARK_MODEL="doubao-seed-2-1-turbo-260628",
    )

    assert settings.resolved_chat_model == "doubao-seed-2-1-turbo-260628"


def test_openviking_memory_defaults_to_isolated_virtual_girlfriend_port() -> None:
    settings = Settings(_env_file=None)

    assert settings.memory_enabled is False
    assert settings.memory_base_url == "http://127.0.0.1:1934"
    assert settings.memory_agent_id == "virtual-girlfriend"
    assert settings.memory_commit_every_turns == 1


def test_offline_avatar_media_proxy_must_target_loopback() -> None:
    with pytest.raises(ValidationError, match="VH_AVATAR_MEDIA_BASE_URL"):
        Settings(
            _env_file=None,
            VH_OFFLINE_RUNTIME=True,
            VH_UPSTREAM_MODE="omlx",
            VH_CHAT_BACKEND="omlx",
            VH_AVATAR_MEDIA_BASE_URL="https://renderer.example",
        )


def test_avatar_renderer_token_is_read_from_a_separate_file(tmp_path) -> None:
    token_file = tmp_path / "renderer.token"
    token_file.write_text("private-renderer-token\n")
    settings = Settings(
        _env_file=None,
        VH_AVATAR_RENDERER_TOKEN_FILE=str(token_file),
    )

    assert settings.resolved_avatar_renderer_token() == "private-renderer-token"


def test_locked_avatar_rejects_a_different_reference() -> None:
    with pytest.raises(ValidationError, match="identity is locked"):
        Settings(
            _env_file=None,
            VH_AVATAR_REFERENCE="public/avatar-reference.jpg",
        )


def test_locked_voice_rejects_unapproved_or_noisy_reference() -> None:
    with pytest.raises(ValidationError, match="Voice profile is locked"):
        Settings(
            _env_file=None,
            VH_TTS_BASE_URL="http://127.0.0.1:8010/v1",
            VH_OMLX_TTS_MODEL="Higgs-TTS-3-4B-bf16",
            VH_OMLX_TTS_REF_AUDIO="noisy.wav",
            VH_OMLX_TTS_REF_TEXT="参考台词。",
        )


def test_locked_voice_allows_only_the_service_endpoint_to_move() -> None:
    settings = Settings(
        _env_file=None,
        VH_TTS_BASE_URL="https://private-higgs.example/v1",
    )

    assert settings.resolved_tts_base_url == "https://private-higgs.example/v1"


def test_locked_voice_allows_the_verified_reference_on_same_host() -> None:
    settings = Settings(
        _env_file=None,
        VH_OMLX_TTS_REF_AUDIO=str(LOCKED_VOICE_REFERENCE),
        VH_OMLX_TTS_REF_TEXT=LOCKED_VOICE_REFERENCE_TEXT,
        VH_TTS_PROTOCOL="vllm_omni_higgs",
    )

    assert settings.omlx_tts_ref_text == LOCKED_VOICE_REFERENCE_TEXT
    assert settings.tts_protocol == "vllm_omni_higgs"


def test_offline_runtime_accepts_only_loopback_model_services() -> None:
    settings = Settings(
        _env_file=None,
        VH_VOICE_PROFILE_LOCKED=False,
        VH_OFFLINE_RUNTIME=True,
        VH_UPSTREAM_MODE="omlx",
        VH_CHAT_BACKEND="omlx",
        VH_OMLX_BASE_URL="http://127.0.0.1:8000/v1",
        VH_ASR_BASE_URL="http://127.0.0.1:8001/v1",
        VH_TTS_BASE_URL="http://127.0.0.1:8010/v1",
        VH_MEMORY_ENABLED=True,
        VH_MEMORY_BASE_URL="http://127.0.0.1:1934",
        VH_AVATAR_RENDERER_WS="ws://127.0.0.1:8770/avatar",
    )

    assert settings.offline_runtime is True
    assert settings.resolved_asr_base_url == "http://127.0.0.1:8001/v1"


def test_offline_runtime_rejects_external_inference_endpoint() -> None:
    with pytest.raises(ValidationError, match="loopback-only VH_ASR_BASE_URL"):
        Settings(
            _env_file=None,
            VH_VOICE_PROFILE_LOCKED=False,
            VH_OFFLINE_RUNTIME=True,
            VH_UPSTREAM_MODE="omlx",
            VH_CHAT_BACKEND="omlx",
            VH_OMLX_BASE_URL="http://127.0.0.1:8000/v1",
            VH_ASR_BASE_URL="https://speech.example/v1",
            VH_TTS_BASE_URL="http://127.0.0.1:8010/v1",
        )
