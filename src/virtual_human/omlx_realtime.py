from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import re
import struct
import wave
from collections.abc import AsyncIterator
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from virtual_human.config import Settings
from virtual_human.memory import OpenVikingMemory


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:18]}"


def _pcm16_wav(pcm16: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16)
    return output.getvalue()


def _parse_streaming_wav_header(data: bytes) -> tuple[int, int, int, int] | None:
    """Return (PCM offset, sample rate, channels, sample width) once complete."""
    if len(data) < 12:
        return None
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("oMLX TTS did not return a RIFF/WAVE stream")

    offset = 12
    audio_format: tuple[int, int, int] | None = None
    while len(data) >= offset + 8:
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        payload_offset = offset + 8
        if chunk_id == b"fmt ":
            if len(data) < payload_offset + chunk_size:
                return None
            if chunk_size < 16:
                raise ValueError("oMLX TTS returned an invalid WAV fmt chunk")
            encoding, channels, sample_rate, _, _, bits_per_sample = struct.unpack_from(
                "<HHIIHH", data, payload_offset
            )
            if encoding != 1 or bits_per_sample != 16:
                raise ValueError("oMLX TTS streaming requires PCM16 WAV audio")
            audio_format = (sample_rate, channels, bits_per_sample // 8)
        elif chunk_id == b"data":
            if audio_format is None:
                raise ValueError("oMLX TTS WAV stream is missing its fmt chunk")
            return payload_offset, *audio_format

        padded_size = chunk_size + (chunk_size % 2)
        if len(data) < payload_offset + padded_size:
            return None
        offset = payload_offset + padded_size
    return None


def _clean_assistant_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\s+", "", text).strip()
    sentences = [
        sentence.strip()
        for sentence in re.findall(r".*?[。！？!?](?:[”’」』])?|.+$", text)
        if sentence.strip()
    ]
    if len(text) <= 72 and len(sentences) <= 4:
        return text

    spoken = ""
    for sentence in sentences[:4]:
        if len(spoken) + len(sentence) > 72:
            break
        spoken += sentence
    if spoken:
        return spoken
    return text[:71].rstrip("，,；;：: ") + "。"


def _ark_response_text(payload: dict[str, Any]) -> str:
    direct = payload.get("content")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    fragments: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if content.get("type") == "output_text" and isinstance(text, str):
                fragments.append(text)
    return "".join(fragments).strip()


def _prepare_g_continuous_text(text: str) -> str:
    """Remove theatrical controls while preserving conversational intent.

    G-continuous relies on natural wording instead of per-sentence emotion tags
    or forced pauses.  Keep question/exclamation intent, but remove ellipses and
    join very short sentence fragments that commonly trigger a fresh TTS take.
    """
    text = re.sub(
        r"<\|(?:emotion|prosody|style|sfx):[^|>]+\|>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?:\.{2,}|…+)", "，", text)
    text = re.sub(r"[；;：:]+", "，", text)
    text = re.sub(
        r"([^，。！？!?]{1,5})。(?=(?:不过|但是|可是|然后|所以|好啦|那|先|现在))",
        r"\1，",
        text,
    )
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"，{2,}", "，", text).strip("，")
    if text and text[-1] not in "。！？!?":
        text += "。"
    return text


def _tts_token_budget(text: str) -> int:
    """Bound runaway Higgs generations without changing the locked voice.

    One Higgs audio token is approximately 40 ms. Chinese dialogue normally
    needs up to roughly ten tokens per spoken character once natural pauses
    and expressive delivery are included.  Keep enough headroom for the final
    sentence while still preventing a short reply from turning into a
    40-second stream when the model misses its end token.
    """
    spoken_characters = len(re.sub(r"[\s，。！？!?、]+", "", text))
    return min(640, max(128, spoken_characters * 10 + 48))


def _split_tts_segments(text: str) -> list[str]:
    """Split only at natural sentence endings for reliable Higgs synthesis.

    Higgs can occasionally stop voicing after the first sentence while still
    emitting several seconds of near-silent audio.  Independent requests keep
    every sentence conditioned on its own text, and concatenating their PCM
    preserves the original wording without introducing artificial pauses.
    """
    segments = [
        match.group(0).strip()
        for match in re.finditer(r".+?(?:[。！？!?]+|$)", text)
        if match.group(0).strip()
    ]
    return segments or ([text] if text else [])


def _compact_long_pcm_silence(pcm16: bytes, sample_rate: int) -> bytes:
    """Shorten codec padding and unnaturally long pauses without touching speech.

    Higgs often adds about 150-250 ms of silence to both sides of an
    independently synthesized sentence. Two adjacent sentences can therefore
    sound like the browser stopped even though every PCM chunk arrived. Work in
    10 ms windows so cuts happen only inside sustained near-silence.
    """
    if not pcm16 or len(pcm16) % 2 or sample_rate <= 0:
        return pcm16

    frame_samples = max(1, round(sample_rate * 0.01))
    frame_bytes = frame_samples * 2
    frames = [
        pcm16[offset : offset + frame_bytes]
        for offset in range(0, len(pcm16), frame_bytes)
    ]

    def is_silent(frame: bytes) -> bool:
        sample_count = len(frame) // 2
        if not sample_count:
            return True
        samples = struct.unpack(f"<{sample_count}h", frame[: sample_count * 2])
        rms = math.isqrt(sum(sample * sample for sample in samples) // sample_count)
        return rms <= 180

    silent = [is_silent(frame) for frame in frames]
    keep = [True] * len(frames)
    cursor = 0
    while cursor < len(frames):
        if not silent[cursor]:
            cursor += 1
            continue
        end = cursor + 1
        while end < len(frames) and silent[end]:
            end += 1

        run_frames = end - cursor
        at_edge = cursor == 0 or end == len(frames)
        minimum_frames = 8 if at_edge else 25
        target_frames = 5 if at_edge else 14
        if run_frames > minimum_frames:
            drop = run_frames - target_frames
            drop_start = cursor + target_frames // 2
            for index in range(drop_start, drop_start + drop):
                keep[index] = False
        cursor = end

    return b"".join(frame for frame, retained in zip(frames, keep) if retained)


def _normalized_spoken_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def _looks_like_assistant_echo(candidate: str, assistant: str) -> bool:
    """Reject playback echo while allowing an explicit, genuine interruption."""
    candidate_normalized = _normalized_spoken_text(candidate)
    assistant_normalized = _normalized_spoken_text(assistant)
    if not candidate_normalized:
        return True
    if candidate_normalized in {"停", "等等", "等一下", "别说了", "打住"}:
        return False
    if len(candidate_normalized) <= 2:
        return True
    if not assistant_normalized:
        return False
    if candidate_normalized in assistant_normalized:
        return True
    matcher = SequenceMatcher(
        None,
        candidate_normalized,
        assistant_normalized,
        autojunk=False,
    )
    longest = matcher.find_longest_match()
    # Speaker playback is frequently transcribed with several missing or
    # substituted syllables. Compare both the whole phrase and its longest
    # preserved span so imperfect echo cannot cut an answer short.
    return (
        matcher.ratio() >= 0.66
        or longest.size / len(candidate_normalized) >= 0.66
    )


class OMLXRealtimeSession:
    """Push-to-talk Realtime adapter backed entirely by local oMLX REST APIs."""

    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self.sample_rate = settings.output_sample_rate
        self.session_id = _id("sess")
        self._events: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._response_task: asyncio.Task[None] | None = None
        self._input_audio = bytearray()
        self._barge_input_audio = bytearray()
        self._last_assistant_text = ""
        self._history: list[dict[str, str]] = []
        self._instructions = settings.persona_instructions
        self._memory_turns_since_commit = 0
        self._memory: OpenVikingMemory | None = None
        self._memory_session_enabled = False
        if settings.memory_enabled:
            self._memory = OpenVikingMemory(
                settings.memory_base_url,
                api_key=settings.memory_api_key,
                timeout_seconds=settings.memory_timeout_seconds,
                search_limit=settings.memory_search_limit,
                score_threshold=settings.memory_score_threshold,
                agent_id=settings.memory_agent_id,
                transport=transport,
            )
            self._memory_session_enabled = True
        self._tts_ref_audio: str | None = None
        self._tts_reference_path: str | None = None
        if settings.omlx_tts_ref_audio:
            reference_path = Path(settings.omlx_tts_ref_audio).expanduser()
            if not reference_path.is_file():
                raise FileNotFoundError(f"TTS reference audio not found: {reference_path}")
            self._tts_reference_path = str(reference_path.resolve())
            if settings.tts_protocol in {"openai", "vllm_omni_higgs"}:
                self._tts_ref_audio = base64.b64encode(reference_path.read_bytes()).decode("ascii")
        api_key = settings.resolved_omlx_api_key()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=settings.omlx_base_url.rstrip("/") + "/",
            headers=headers,
            timeout=httpx.Timeout(180),
            transport=transport,
        )
        self._asr_client = httpx.AsyncClient(
            base_url=settings.resolved_asr_base_url.rstrip("/") + "/",
            headers=headers,
            timeout=httpx.Timeout(180),
            transport=transport,
        )
        self._tts_client = httpx.AsyncClient(
            base_url=settings.resolved_tts_base_url.rstrip("/") + "/",
            headers=headers,
            timeout=httpx.Timeout(180),
            transport=transport,
        )
        self._chat_client = self._client
        if settings.chat_backend == "ark":
            self._chat_client = httpx.AsyncClient(
                base_url=settings.ark_base_url.rstrip("/") + "/",
                headers={"Authorization": f"Bearer {settings.ark_api_key}"},
                timeout=httpx.Timeout(180),
                transport=transport,
            )

    async def start(self) -> None:
        await self._events.put(
            {
                "type": "session.created",
                "event_id": _id("event"),
                "session": {
                    "id": self.session_id,
                    "type": "realtime",
                    "model": self.settings.omlx_chat_model,
                    "audio": {
                        "input": {"format": {"type": "audio/pcm", "rate": self.sample_rate}},
                        "output": {
                            "format": {"type": "audio/pcm", "rate": self.sample_rate},
                            "voice": self.settings.omlx_tts_voice,
                        },
                    },
                    "memory": {"enabled": self._memory_session_enabled},
                },
            }
        )

    async def send(self, raw: str) -> None:
        event = json.loads(raw)
        event_type = event.get("type")
        if event_type == "session.update":
            session = event.get("session") or {}
            if session.get("instructions"):
                self._instructions = str(session["instructions"])
            memory = session.get("memory")
            if isinstance(memory, dict) and isinstance(memory.get("enabled"), bool):
                self._memory_session_enabled = bool(memory["enabled"] and self._memory)
            await self._events.put(
                {
                    "type": "session.updated",
                    "event_id": _id("event"),
                    "session": session,
                }
            )
        elif event_type == "input_audio_buffer.append":
            audio = event.get("audio")
            if audio:
                self._input_audio.extend(base64.b64decode(audio, validate=True))
        elif event_type == "input_audio_buffer.clear":
            self._input_audio.clear()
            self._barge_input_audio.clear()
        elif event_type == "input_audio_buffer.commit":
            await self._start_response(from_audio=True)
        elif event_type == "input_audio_buffer.barge_in.append":
            if not self.settings.barge_in_enabled:
                return
            audio = event.get("audio")
            if audio:
                self._barge_input_audio.extend(base64.b64decode(audio, validate=True))
        elif event_type == "input_audio_buffer.barge_in.commit":
            if not self.settings.barge_in_enabled:
                self._barge_input_audio.clear()
                return
            await self._handle_barge_in()
        elif event_type == "response.create":
            await self._start_response(from_audio=bool(self._input_audio))
        elif event_type == "response.cancel":
            await self.cancel()

    async def _start_response(self, *, from_audio: bool) -> None:
        if self._response_task and not self._response_task.done():
            return
        pcm16 = bytes(self._input_audio)
        self._input_audio.clear()
        self._response_task = asyncio.create_task(self._respond(pcm16 if from_audio else b""))

    async def _handle_barge_in(self) -> None:
        pcm16 = bytes(self._barge_input_audio)
        self._barge_input_audio.clear()
        duration_seconds = len(pcm16) / (self.sample_rate * 2)
        if duration_seconds < 0.25:
            await self._events.put(
                {
                    "type": "input_audio_buffer.barge_in.echo_ignored",
                    "event_id": _id("event"),
                    "reason": "too_short",
                }
            )
            return

        user_text = await self._transcribe(pcm16)
        normalized = _normalized_spoken_text(user_text)
        if (
            not normalized
            or re.fullmatch(r"[嗯啊呃哦唔诶哎]+", normalized)
            or _looks_like_assistant_echo(user_text, self._last_assistant_text)
        ):
            await self._events.put(
                {
                    "type": "input_audio_buffer.barge_in.echo_ignored",
                    "event_id": _id("event"),
                    "reason": "playback_echo",
                }
            )
            return

        await self.cancel()
        await self._events.put(
            {
                "type": "input_audio_buffer.barge_in.accepted",
                "event_id": _id("event"),
                "transcript": user_text,
            }
        )
        self._response_task = asyncio.create_task(
            self._respond(b"", user_text_override=user_text)
        )

    async def _transcribe(self, pcm16: bytes) -> str:
        response = await self._asr_client.post(
            "audio/transcriptions",
            files={"file": ("input.wav", _pcm16_wav(pcm16, self.sample_rate), "audio/wav")},
            data={
                "model": self.settings.omlx_stt_model,
                "language": "zh",
                "response_format": "json",
            },
        )
        response.raise_for_status()
        return str(response.json().get("text") or "").strip()

    async def _chat(self, user_text: str) -> str:
        memory_context = (
            await self._memory.recall(user_text)
            if self._memory and self._memory_session_enabled
            else ""
        )
        instructions = self._instructions
        if memory_context:
            instructions += (
                "\n\n下面是虚拟女友专用记忆库中与本轮相关的已知事实。"
                "只在确实相关时自然使用，不要逐条复述，不要提到“记忆库”或检索过程，"
                "不要把推测当事实。标为“长期实体”或“稳定偏好”的用户明确事实，"
                "优先于“历史事件”中助手自己说过的旧回答；历史助手回答可能是当时的误判，"
                "不能用来否定用户已经明确告知的事实。回答姓名、年龄、关系或身份问题时，"
                "“用户资料·最高优先”是用户本人的确定资料，必须逐字采用；"
                "只要它或“长期实体”已明确给出答案，就必须直接采用，不能回答不知道；"
                "历史事件里的助手话语不属于事实：\n"
                f"{memory_context}"
            )
        conversation = [
            *self._history[-12:],
            {"role": "user", "content": user_text},
        ]
        if self.settings.chat_backend == "ark":
            response = await self._chat_client.post(
                "responses",
                json={
                    "model": self.settings.ark_model,
                    "instructions": instructions,
                    "input": conversation,
                    "temperature": self.settings.omlx_chat_temperature,
                    "top_p": self.settings.omlx_chat_top_p,
                    "max_output_tokens": self.settings.omlx_chat_max_tokens,
                    "thinking": {"type": "disabled"},
                    "store": False,
                },
            )
        else:
            response = await self._chat_client.post(
                "chat/completions",
                json={
                    "model": self.settings.omlx_chat_model,
                    "messages": [
                        {"role": "system", "content": instructions},
                        *conversation,
                    ],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "temperature": self.settings.omlx_chat_temperature,
                    "top_p": self.settings.omlx_chat_top_p,
                    "max_tokens": self.settings.omlx_chat_max_tokens,
                },
            )
        response.raise_for_status()
        payload = response.json()
        if self.settings.chat_backend == "ark":
            raw_text = _ark_response_text(payload)
        else:
            raw_text = str(payload["choices"][0]["message"]["content"])
        text = _clean_assistant_text(raw_text)
        if not text:
            raise RuntimeError(f"{self.settings.chat_backend} chat model returned an empty response")
        self._history.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": text},
            ]
        )
        if (
            self._memory
            and self._memory_session_enabled
            and await self._memory.record_turn(self.session_id, user_text, text)
        ):
            self._memory_turns_since_commit += 1
            if self._memory_turns_since_commit >= self.settings.memory_commit_every_turns:
                if await self._memory.commit(self.session_id):
                    self._memory_turns_since_commit = 0
        return text

    async def _synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield 40 ms PCM16 chunks from oMLX's native streaming TTS endpoint."""
        spoken_text = (
            _prepare_g_continuous_text(text)
            if self.settings.tts_flow_mode == "g_continuous"
            else text
        )
        chunk_bytes = max(2, round(self.sample_rate * 0.04) * 2)
        for spoken_segment in _split_tts_segments(spoken_text):
            if self.settings.tts_protocol == "sglang_higgs":
                request = {
                    "input": spoken_segment,
                    "temperature": self.settings.omlx_tts_temperature,
                    "top_p": self.settings.omlx_tts_top_p,
                    "max_new_tokens": _tts_token_budget(spoken_segment),
                    # SGLang-Omni returns a single canonical WAV in this mode.
                    # Its streaming mode uses SSE rather than raw WAV bytes.
                    "stream": False,
                }
                if self.settings.tts_top_k is not None:
                    request["top_k"] = self.settings.tts_top_k
                if self._tts_reference_path is not None:
                    request["references"] = [
                        {
                            "audio_path": self._tts_reference_path,
                            "text": self.settings.omlx_tts_ref_text,
                        }
                    ]
            elif self.settings.tts_protocol == "vllm_omni_higgs":
                request = {
                    "model": self.settings.tts_served_model,
                    "input": spoken_segment,
                    "response_format": "wav",
                    "max_new_tokens": _tts_token_budget(spoken_segment),
                    "temperature": self.settings.omlx_tts_temperature,
                    "top_p": self.settings.omlx_tts_top_p,
                    "stream": False,
                }
                if self.settings.tts_seed is not None:
                    request["seed"] = self.settings.tts_seed
                if self.settings.tts_top_k is not None:
                    request["top_k"] = self.settings.tts_top_k
                if self._tts_ref_audio is not None:
                    request["ref_audio"] = f"data:audio/wav;base64,{self._tts_ref_audio}"
                    request["ref_text"] = self.settings.omlx_tts_ref_text
            else:
                request = {
                    "model": self.settings.omlx_tts_model,
                    "input": spoken_segment,
                    "language": "Chinese",
                    "response_format": "wav",
                    "speed": 1.0,
                    "temperature": self.settings.omlx_tts_temperature,
                    "top_p": self.settings.omlx_tts_top_p,
                    "repetition_penalty": self.settings.omlx_tts_repetition_penalty,
                    "stream": True,
                    "streaming_interval": self.settings.tts_streaming_interval,
                    "max_tokens": _tts_token_budget(spoken_segment),
                }
                if self.settings.tts_streaming_mode != "full":
                    request["streaming_mode"] = self.settings.tts_streaming_mode
                if self.settings.omlx_tts_voice:
                    request["voice"] = self.settings.omlx_tts_voice
                if self.settings.omlx_tts_instructions:
                    request["instructions"] = self.settings.omlx_tts_instructions
                if self.settings.tts_seed is not None:
                    request["seed"] = self.settings.tts_seed
                if self.settings.tts_top_k is not None:
                    request["top_k"] = self.settings.tts_top_k
                if self._tts_ref_audio is not None:
                    request["ref_audio"] = self._tts_ref_audio
                    request["ref_text"] = self.settings.omlx_tts_ref_text

            header_buffer = bytearray()
            pcm_buffer = bytearray()
            header_parsed = False
            async with self._tts_client.stream(
                "POST",
                "audio/speech",
                json=request,
            ) as response:
                response.raise_for_status()
                async for network_chunk in response.aiter_bytes():
                    if not header_parsed:
                        header_buffer.extend(network_chunk)
                        parsed = _parse_streaming_wav_header(header_buffer)
                        if parsed is None:
                            continue
                        pcm_offset, source_rate, channels, sample_width = parsed
                        if (source_rate, channels, sample_width) != (
                            self.sample_rate,
                            1,
                            2,
                        ):
                            raise ValueError(
                                "oMLX TTS stream must be 24kHz mono PCM16; "
                                f"received {source_rate}Hz/{channels}ch/"
                                f"{sample_width * 8}bit"
                            )
                        pcm_buffer.extend(header_buffer[pcm_offset:])
                        header_buffer.clear()
                        header_parsed = True
                    else:
                        pcm_buffer.extend(network_chunk)

                    if not self.settings.omlx_tts_model.lower().startswith("higgs"):
                        while len(pcm_buffer) >= chunk_bytes:
                            yield bytes(pcm_buffer[:chunk_bytes])
                            del pcm_buffer[:chunk_bytes]

            if not header_parsed:
                raise ValueError("oMLX TTS returned an incomplete WAV stream")
            if len(pcm_buffer) % 2:
                pcm_buffer.pop()
            segment_pcm = bytes(pcm_buffer)
            if self.settings.omlx_tts_model.lower().startswith("higgs"):
                segment_pcm = _compact_long_pcm_silence(
                    segment_pcm,
                    self.sample_rate,
                )
            for offset in range(0, len(segment_pcm), chunk_bytes):
                yield segment_pcm[offset : offset + chunk_bytes]

    async def _respond(
        self,
        pcm16: bytes,
        *,
        user_text_override: str | None = None,
    ) -> None:
        response_id = _id("resp")
        item_id = _id("item")
        try:
            self._last_assistant_text = ""
            if user_text_override is not None:
                user_text = user_text_override
                await self._events.put(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "event_id": _id("event"),
                        "transcript": user_text,
                    }
                )
            elif pcm16:
                await self._events.put(
                    {"type": "input_audio_buffer.speech_started", "event_id": _id("event")}
                )
                user_text = await self._transcribe(pcm16)
                await self._events.put(
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "event_id": _id("event"),
                        "transcript": user_text,
                    }
                )
                normalized = re.sub(r"[\s，。！？!?、]+", "", user_text)
                duration_seconds = len(pcm16) / (self.sample_rate * 2)
                if not normalized or (
                    duration_seconds < 1.0 and re.fullmatch(r"[嗯啊呃哦唔诶哎]+", normalized)
                ):
                    await self._events.put(
                        {
                            "type": "input_audio_buffer.no_speech",
                            "event_id": _id("event"),
                        }
                    )
                    return
            else:
                user_text = "请自然地和我打个招呼。"

            await self._events.put(
                {
                    "type": "response.created",
                    "event_id": _id("event"),
                    "response": {"id": response_id, "status": "in_progress", "output": []},
                }
            )
            text = await self._chat(user_text)
            self._last_assistant_text = text
            spoken_text = (
                _prepare_g_continuous_text(text)
                if self.settings.tts_flow_mode == "g_continuous"
                else text
            )
            await self._events.put(
                {
                    "type": "response.output_audio_transcript.done",
                    "event_id": _id("event"),
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    # The caption must be the exact text sent to TTS.  Showing
                    # the pre-prosody LLM text makes punctuation, pauses and
                    # stripped control tags visibly disagree with the voice.
                    "transcript": spoken_text,
                }
            )
            async for pcm_chunk in self._synthesize_stream(spoken_text):
                await self._events.put(
                    {
                        "type": "response.output_audio.delta",
                        "event_id": _id("event"),
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": base64.b64encode(pcm_chunk).decode("ascii"),
                    }
                )
                await asyncio.sleep(0)
            await self._events.put(
                {
                    "type": "response.output_audio.done",
                    "event_id": _id("event"),
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                }
            )
            await self._events.put(
                {
                    "type": "response.done",
                    "event_id": _id("event"),
                    "response": {"id": response_id, "status": "completed", "output": []},
                }
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._events.put(
                {
                    "type": "error",
                    "event_id": _id("event"),
                    "error": {"type": "omlx_error", "message": str(error)},
                }
            )

    async def cancel(self) -> None:
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass
        self._response_task = None

    async def events(self) -> AsyncIterator[str]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield json.dumps(event, ensure_ascii=False)

    async def close(self) -> None:
        await self.cancel()
        if self._memory:
            if self._memory_session_enabled and self._memory_turns_since_commit:
                await self._memory.commit(self.session_id)
            await self._memory.close()
        if self._chat_client is not self._client:
            await self._chat_client.aclose()
        await self._asr_client.aclose()
        await self._client.aclose()
        await self._tts_client.aclose()
        await self._events.put(None)
