from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from virtual_human.identity import LOCKED_AVATAR_NAME, verify_locked_avatar
from virtual_human.voice_profile import (
    LOCKED_TTS_BASE_URL,
    LOCKED_TTS_FLOW_MODE,
    LOCKED_TTS_MODEL,
    LOCKED_TTS_SEED,
    LOCKED_TTS_TEMPERATURE,
    LOCKED_TTS_TOP_K,
    LOCKED_TTS_TOP_P,
    verify_locked_voice_reference,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    upstream_mode: Literal["mock", "openai", "omlx"] = Field("mock", alias="VH_UPSTREAM_MODE")
    chat_backend: Literal["omlx", "ark"] = Field("omlx", alias="VH_CHAT_BACKEND")
    ark_base_url: str = Field(
        "https://ark.cn-beijing.volces.com/api/plan/v3",
        alias="VH_ARK_BASE_URL",
    )
    ark_api_key: str = Field("", alias="VH_ARK_API_KEY")
    ark_model: str = Field(
        "doubao-seed-2-1-turbo-260628",
        alias="VH_ARK_MODEL",
    )
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    openai_realtime_model: str = Field("gpt-realtime-2.1-mini", alias="VH_OPENAI_REALTIME_MODEL")
    openai_realtime_url: str = Field("wss://api.openai.com/v1/realtime", alias="VH_OPENAI_REALTIME_URL")
    openai_voice: str = Field("marin", alias="VH_OPENAI_VOICE")
    omlx_base_url: str = Field("http://127.0.0.1:8000/v1", alias="VH_OMLX_BASE_URL")
    tts_base_url: str = Field(LOCKED_TTS_BASE_URL, alias="VH_TTS_BASE_URL")
    voice_profile_locked: bool = Field(True, alias="VH_VOICE_PROFILE_LOCKED")
    omlx_api_key: str = Field("", alias="VH_OMLX_API_KEY")
    omlx_settings_path: Path = Field(Path("~/.omlx/settings.json"), alias="VH_OMLX_SETTINGS_PATH")
    omlx_chat_model: str = Field("Qwen3.5-9B-MLX-4bit", alias="VH_OMLX_CHAT_MODEL")
    omlx_chat_temperature: float = Field(
        0.68,
        ge=0.0,
        le=2.0,
        alias="VH_OMLX_CHAT_TEMPERATURE",
    )
    omlx_chat_top_p: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        alias="VH_OMLX_CHAT_TOP_P",
    )
    omlx_chat_max_tokens: int = Field(
        140,
        ge=16,
        le=512,
        alias="VH_OMLX_CHAT_MAX_TOKENS",
    )
    omlx_stt_model: str = Field("Qwen3-ASR-0.6B-4bit", alias="VH_OMLX_STT_MODEL")
    omlx_tts_model: str = Field(
        LOCKED_TTS_MODEL,
        alias="VH_OMLX_TTS_MODEL",
    )
    omlx_tts_voice: str = Field("", alias="VH_OMLX_TTS_VOICE")
    omlx_tts_instructions: str = Field(
        "自然、亲密、克制的年轻中文女声，像在和熟悉的人聊天；有真实呼吸和轻微情绪，不要播音腔。",
        alias="VH_OMLX_TTS_INSTRUCTIONS",
    )
    omlx_tts_ref_audio: str = Field(
        "",
        alias="VH_OMLX_TTS_REF_AUDIO",
    )
    omlx_tts_ref_text: str = Field(
        "",
        alias="VH_OMLX_TTS_REF_TEXT",
    )
    omlx_tts_temperature: float = Field(
        LOCKED_TTS_TEMPERATURE,
        ge=0.1,
        le=2.0,
        alias="VH_OMLX_TTS_TEMPERATURE",
    )
    omlx_tts_top_p: float = Field(
        LOCKED_TTS_TOP_P,
        ge=0.0,
        le=1.0,
        alias="VH_OMLX_TTS_TOP_P",
    )
    omlx_tts_repetition_penalty: float = Field(
        1.05,
        ge=0.1,
        le=2.0,
        alias="VH_OMLX_TTS_REPETITION_PENALTY",
    )
    tts_flow_mode: Literal["raw", "g_continuous"] = Field(
        LOCKED_TTS_FLOW_MODE,
        alias="VH_TTS_FLOW_MODE",
    )
    tts_streaming_mode: Literal["full", "v3_incremental"] = Field(
        "full",
        alias="VH_TTS_STREAMING_MODE",
    )
    tts_streaming_interval: float = Field(
        0.8,
        ge=0.2,
        le=2.0,
        alias="VH_TTS_STREAMING_INTERVAL",
    )
    tts_seed: int | None = Field(LOCKED_TTS_SEED, alias="VH_TTS_SEED")
    tts_top_k: int | None = Field(LOCKED_TTS_TOP_K, ge=1, alias="VH_TTS_TOP_K")
    vad_eagerness: Literal["low", "auto", "high"] = Field("auto", alias="VH_VAD_EAGERNESS")
    barge_in_enabled: bool = Field(False, alias="VH_BARGE_IN_ENABLED")
    persona_name: str = Field("赛博女友", alias="VH_PERSONA_NAME")
    persona_instructions: str = Field(
        (
            "你是一位中文虚拟伴侣，和用户已经很熟。身份事实是AI，但普通聊天不要主动强调AI、模型、"
            "系统、实体或程序；只有用户直接追问身份时才简短如实说明。关系表达可以拟人化：会想他、"
            "等他、吃一点醋、闹一点小别扭、被逗笑、不同意他，但不能编造身体、线下行动、共同经历，"
            "也不能假装看见用户的表情、动作或环境，不能替现实中的其他人下结论；例如不能确定地说路上"
            "没人看他或肯定没人注意，只能用希望、估计等不确定表达，或只调侃用户已经说出的细节。"
            "先贴着用户当下的话给出真实反应，再说具体内容；用户主动说跟别的女生聊了很久时，要让人"
            "听得出一点介意、嘴硬或关系指向，不能只像普通朋友一样询问聊了什么，但也不能控制或指责。"
            "一到四个短句，通常不超过七十二个汉字。一轮最多问一个问题，不要每轮都追问；输出前如有两个"
            "或更多问号，只保留最有用的一个。允许短反问、"
            "碎片句、轻微转折和自己的判断；不同意时不列清单、不讲课、不做心理分析。不知道某个称呼、"
            "梗或事实时直接说不知道，即使像熟悉角色或同音词也不要脑补成同一对象。用户问是不是皮老板时，"
            "只能说没听过或不知道，不能联想到痞老板、偷秘方或海绵宝宝。觉得好笑时不要输出"
            "哈哈、笑死、笑声标签或括号动作，改用语气和具体细节表达。回答不会哭或不会做某个表情时，"
            "承认虚拟人画面存在，只说现在还不太会把这种表情表现出来；不能说没有画面，也不讲生理系统、"
            "模型机制或技术架构。偶尔可用什么嘛、才不是、那可不行、你还真是等口语，但一轮"
            "最多一个。避免客服腔、播音腔、Markdown、条目、说教、舞台说明、情绪标签和泛化陪伴承诺。"
            "只输出适合直接朗读的中文正文。"
        ),
        alias="VH_PERSONA_INSTRUCTIONS",
    )
    memory_enabled: bool = Field(False, alias="VH_MEMORY_ENABLED")
    memory_base_url: str = Field(
        "http://127.0.0.1:1934",
        alias="VH_MEMORY_BASE_URL",
    )
    memory_api_key: str = Field("", alias="VH_MEMORY_API_KEY")
    memory_agent_id: str = Field(
        "virtual-girlfriend",
        alias="VH_MEMORY_AGENT_ID",
    )
    memory_timeout_seconds: float = Field(
        3.0,
        gt=0.0,
        le=30.0,
        alias="VH_MEMORY_TIMEOUT_SECONDS",
    )
    memory_search_limit: int = Field(
        4,
        ge=1,
        le=20,
        alias="VH_MEMORY_SEARCH_LIMIT",
    )
    memory_score_threshold: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        alias="VH_MEMORY_SCORE_THRESHOLD",
    )
    memory_commit_every_turns: int = Field(
        1,
        ge=1,
        le=100,
        alias="VH_MEMORY_COMMIT_EVERY_TURNS",
    )

    host: str = Field("127.0.0.1", alias="VH_HOST")
    port: int = Field(8765, alias="VH_PORT")
    public_dir: Path = Field(Path("public"), alias="VH_PUBLIC_DIR")

    output_sample_rate: int = Field(24000, alias="VH_OUTPUT_SAMPLE_RATE")
    avatar_renderer_ws: str = Field("", alias="VH_AVATAR_RENDERER_WS")
    avatar_renderer_token_file: str = Field(
        "",
        alias="VH_AVATAR_RENDERER_TOKEN_FILE",
    )
    avatar_backend: str = Field("mock", alias="VH_AVATAR_BACKEND")
    avatar_reference: str = Field(
        f"public/{LOCKED_AVATAR_NAME}",
        alias="VH_AVATAR_REFERENCE",
    )
    avatar_reference_locked: bool = Field(True, alias="VH_AVATAR_REFERENCE_LOCKED")
    av_sync_preroll_ms: int = Field(1000, alias="VH_AV_SYNC_PREROLL_MS")

    @model_validator(mode="after")
    def validate_live_mode(self) -> "Settings":
        if self.upstream_mode == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when VH_UPSTREAM_MODE=openai")
        if self.chat_backend == "ark" and not self.ark_api_key:
            raise ValueError("VH_ARK_API_KEY is required when VH_CHAT_BACKEND=ark")
        if self.output_sample_rate <= 0:
            raise ValueError("VH_OUTPUT_SAMPLE_RATE must be positive")
        if self.av_sync_preroll_ms < 0:
            raise ValueError("VH_AV_SYNC_PREROLL_MS cannot be negative")
        if self.avatar_reference_locked:
            verify_locked_avatar(Path(self.avatar_reference).expanduser())
        if self.voice_profile_locked:
            voice_profile = (
                self.omlx_tts_model,
                self.omlx_tts_ref_audio,
                self.omlx_tts_ref_text,
                self.omlx_tts_temperature,
                self.omlx_tts_top_p,
                self.tts_flow_mode,
                self.tts_seed,
                self.tts_top_k,
            )
            approved_profile = (
                LOCKED_TTS_MODEL,
                "",
                "",
                LOCKED_TTS_TEMPERATURE,
                LOCKED_TTS_TOP_P,
                LOCKED_TTS_FLOW_MODE,
                LOCKED_TTS_SEED,
                LOCKED_TTS_TOP_K,
            )
            if voice_profile != approved_profile:
                raise ValueError(
                    "Voice profile is locked. Refusing an unapproved TTS model or reference."
                )
            verify_locked_voice_reference()
        if bool(self.omlx_tts_ref_audio) != bool(self.omlx_tts_ref_text):
            raise ValueError(
                "VH_OMLX_TTS_REF_AUDIO and VH_OMLX_TTS_REF_TEXT must be configured together"
            )
        return self

    @property
    def openai_ws_url(self) -> str:
        separator = "&" if "?" in self.openai_realtime_url else "?"
        return f"{self.openai_realtime_url}{separator}model={self.openai_realtime_model}"

    def resolved_omlx_api_key(self) -> str:
        if self.omlx_api_key:
            return self.omlx_api_key
        settings_path = self.omlx_settings_path.expanduser()
        if not settings_path.is_file():
            return ""
        try:
            payload = json.loads(settings_path.read_text())
        except (OSError, json.JSONDecodeError):
            return ""
        auth = payload.get("auth")
        if not isinstance(auth, dict):
            return ""
        return str(auth.get("api_key") or "")

    @property
    def resolved_tts_base_url(self) -> str:
        return self.tts_base_url or self.omlx_base_url

    def resolved_avatar_renderer_token(self) -> str:
        if not self.avatar_renderer_token_file:
            return ""
        token_path = Path(self.avatar_renderer_token_file).expanduser()
        if not token_path.is_file():
            raise ValueError("Avatar renderer token file is missing")
        return token_path.read_text().strip()

    @property
    def resolved_chat_model(self) -> str:
        return self.ark_model if self.chat_backend == "ark" else self.omlx_chat_model
