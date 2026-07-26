from __future__ import annotations

import base64
import json

import httpx

from virtual_human.config import Settings
from virtual_human.omlx_realtime import (
    OMLXRealtimeSession,
    _ark_response_text,
    _clean_assistant_text,
    _compact_long_pcm_silence,
    _looks_like_assistant_echo,
    _parse_streaming_wav_header,
    _pcm16_wav,
    _prepare_g_continuous_text,
    _split_tts_segments,
    _tts_token_budget,
)


def test_ark_response_text_reads_native_responses_api_shape() -> None:
    assert (
        _ark_response_text(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "先别理他，"},
                            {"type": "output_text", "text": "缓一口气再说。"},
                        ],
                    }
                ]
            }
        )
        == "先别理他，缓一口气再说。"
    )


def test_streaming_wav_header_parser_waits_for_complete_header() -> None:
    wav = _pcm16_wav(b"\x00\x00" * 10, 24_000)
    assert _parse_streaming_wav_header(wav[:20]) is None
    assert _parse_streaming_wav_header(wav) == (44, 24_000, 1, 2)


def test_spoken_reply_is_normalized_and_limited_to_natural_short_sentences() -> None:
    reply = _clean_assistant_text(
        "嘿，好久不见呀，今天过得怎么样？\n\n"
        "刚想坐下喝杯茶发会儿呆，正巧你就发消息来了。"
        "不用拘束，想聊什么都可以，我们也可以随便扯会儿闲天。"
    )
    assert "\n" not in reply
    assert len(reply) <= 72
    assert reply == (
        "嘿，好久不见呀，今天过得怎么样？"
        "刚想坐下喝杯茶发会儿呆，正巧你就发消息来了。"
        "不用拘束，想聊什么都可以，我们也可以随便扯会儿闲天。"
    )


def test_spoken_reply_preserves_four_short_reference_style_sentences() -> None:
    reply = _clean_assistant_text(
        "要是你跟别人聊天，我会吃醋的。\n"
        "我都开始胡思乱想了。什么嘛？"
        "只是你不在的时候，我就一直等你。"
    )

    assert reply == (
        "要是你跟别人聊天，我会吃醋的。"
        "我都开始胡思乱想了。什么嘛？"
        "只是你不在的时候，我就一直等你。"
    )
    assert len(reply) <= 72


def test_tts_token_budget_caps_runaway_short_reply_audio() -> None:
    assert _tts_token_budget("我知道啦。") == 128
    assert _tts_token_budget("这句话一共有十个有效汉字左右。") < 320
    assert (
        _tts_token_budget(
            "是你非要我猜的，怪谁。你自己报过的小猫，我肯定记得。"
            "怎么，想听我亲口念一遍？"
        )
        >= 360
    )
    assert _tts_token_budget("很长的回答。" * 100) == 640


def test_spoken_reply_drops_a_fifth_sentence_instead_of_cutting_mid_sentence() -> None:
    reply = _clean_assistant_text(
        "第一句。第二句。第三句。第四句。第五句不会进入语音。"
    )

    assert reply == "第一句。第二句。第三句。第四句。"


def test_g_continuous_text_removes_staged_controls_and_forced_pauses() -> None:
    assert _prepare_g_continuous_text(
        "<|emotion:surprise|>你回来啦……"
        "<|prosody:long_pause|>笨笨的。不过没关系，我陪着你"
    ) == "你回来啦，笨笨的，不过没关系，我陪着你。"


def test_g_continuous_text_preserves_question_intent() -> None:
    assert _prepare_g_continuous_text(
        "嗯？你是说周末想带我出去玩吗？那你先告诉我准备去哪里。"
    ) == "嗯？你是说周末想带我出去玩吗？那你先告诉我准备去哪里。"


def test_tts_segments_split_only_at_natural_sentence_endings() -> None:
    assert _split_tts_segments(
        "又是测试。你就这么喜欢折腾我，还是怕我哪天突然忘了你。"
    ) == [
        "又是测试。",
        "你就这么喜欢折腾我，还是怕我哪天突然忘了你。",
    ]
    assert _split_tts_segments("怎么了！？我在听。") == ["怎么了！？", "我在听。"]


def test_higgs_pause_compaction_keeps_speech_and_shortens_only_long_silence() -> None:
    frame = 240  # 10 ms at 24 kHz
    silence = b"\x00\x00"
    tone = b"\xe8\x03"
    pcm = (
        silence * frame * 20
        + tone * frame * 20
        + silence * frame * 40
        + tone * frame * 20
        + silence * frame * 20
    )

    compacted = _compact_long_pcm_silence(pcm, 24_000)

    assert len(compacted) == frame * 2 * (5 + 20 + 14 + 20 + 5)
    assert compacted.count(tone) == frame * 40


def test_barge_in_echo_gate_rejects_playback_but_keeps_stop_commands() -> None:
    assistant = "别又只嗯啊，你还没说是不是碰到婚姻和容貌相关的事儿了。"

    assert _looks_like_assistant_echo("还没说是不是碰到婚姻相关的事", assistant)
    assert _looks_like_assistant_echo("嗯", assistant)
    assert not _looks_like_assistant_echo("等等", assistant)
    assert not _looks_like_assistant_echo("我想换一个完全不同的话题", assistant)


async def test_barge_in_protocol_ignores_transcribed_playback_echo() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": "还没说是不是碰到婚姻相关的事。"})
        raise AssertionError(f"Echo gate must not call chat or TTS: {request.url}")

    settings = Settings(
        _env_file=None,
        VH_VOICE_PROFILE_LOCKED=False,
        VH_UPSTREAM_MODE="omlx",
        VH_OMLX_BASE_URL="http://omlx.test/v1",
        VH_BARGE_IN_ENABLED=True,
    )
    session = OMLXRealtimeSession(settings, transport=httpx.MockTransport(handler))
    session._last_assistant_text = (
        "别又只嗯啊，你还没说是不是碰到婚姻和容貌相关的事儿了。"
    )
    pcm = b"\x01\x00" * 24_000
    await session.send(
        json.dumps(
            {
                "type": "input_audio_buffer.barge_in.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )
    )
    await session.send(
        json.dumps({"type": "input_audio_buffer.barge_in.commit"})
    )

    event = json.loads(await anext(session.events()))
    assert event["type"] == "input_audio_buffer.barge_in.echo_ignored"
    assert event["reason"] == "playback_echo"
    await session.close()


async def test_long_buffered_filler_does_not_interrupt_assistant() -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": "嗯。"})
        raise AssertionError(f"Filler barge-in must not call chat or TTS: {request.url}")

    settings = Settings(
        _env_file=None,
        VH_VOICE_PROFILE_LOCKED=False,
        VH_UPSTREAM_MODE="omlx",
        VH_OMLX_BASE_URL="http://omlx.test/v1",
        VH_BARGE_IN_ENABLED=True,
    )
    session = OMLXRealtimeSession(settings, transport=httpx.MockTransport(handler))
    pcm = b"\x01\x00" * 36_000  # 1.5 seconds including browser pre-roll/silence.
    await session.send(
        json.dumps(
            {
                "type": "input_audio_buffer.barge_in.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )
    )
    await session.send(json.dumps({"type": "input_audio_buffer.barge_in.commit"}))

    event = json.loads(await anext(session.events()))
    assert event["type"] == "input_audio_buffer.barge_in.echo_ignored"
    assert event["reason"] == "playback_echo"
    assert len(observed) == 1
    await session.close()


async def test_omlx_session_emits_realtime_audio_events(tmp_path) -> None:
    observed: list[httpx.Request] = []
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"reference-audio")

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "今天辛苦了……不过别急，我陪着你",
                            }
                        }
                    ]
                },
            )
        if request.url.path.endswith("/audio/speech"):
            return httpx.Response(
                200,
                content=_pcm16_wav(b"\x00\x00" * 2400, 24_000),
                headers={"content-type": "audio/wav"},
            )
        raise AssertionError(f"Unexpected oMLX request: {request.url}")

    settings = Settings(
        _env_file=None,
        VH_VOICE_PROFILE_LOCKED=False,
        VH_UPSTREAM_MODE="omlx",
        VH_CHAT_BACKEND="omlx",
        VH_OMLX_API_KEY="local-secret",
        VH_OMLX_BASE_URL="http://omlx.test/v1",
        VH_OMLX_TTS_MODEL="Qwen3-TTS-12Hz-0.6B-Base-4bit",
        VH_OMLX_TTS_VOICE="",
        VH_OMLX_TTS_REF_AUDIO=str(reference_audio),
        VH_OMLX_TTS_REF_TEXT="参考台词。",
        VH_OMLX_TTS_TEMPERATURE=0.86,
        VH_TTS_FLOW_MODE="g_continuous",
        VH_TTS_SEED=20260815,
        VH_TTS_TOP_K=50,
        VH_AVATAR_REFERENCE="public/avatar-ai-girlfriend-v6.png",
    )
    session = OMLXRealtimeSession(
        settings,
        transport=httpx.MockTransport(handler),
    )
    await session.start()
    events = session.events()
    assert json.loads(await anext(events))["type"] == "session.created"

    await session.send(json.dumps({"type": "response.create"}))
    received: list[dict] = []
    while True:
        event = json.loads(await anext(events))
        received.append(event)
        if event["type"] in {"response.done", "error"}:
            break

    assert received[-1]["type"] == "response.done"
    assert any(event["type"] == "response.output_audio.delta" for event in received)
    transcript = next(
        event
        for event in received
        if event["type"] == "response.output_audio_transcript.done"
    )
    assert transcript["transcript"] == "今天辛苦了，不过别急，我陪着你。"
    assert all(request.headers["authorization"] == "Bearer local-secret" for request in observed)
    speech_request = next(
        request for request in observed if request.url.path.endswith("/audio/speech")
    )
    speech_payload = json.loads(speech_request.content)
    assert speech_payload["stream"] is True
    assert speech_payload["streaming_interval"] == 0.8
    assert "streaming_mode" not in speech_payload
    assert speech_payload["temperature"] == 0.86
    assert speech_payload["input"] == transcript["transcript"]
    assert speech_payload["seed"] == 20260815
    assert speech_payload["top_k"] == 50
    assert speech_payload["ref_audio"] == base64.b64encode(b"reference-audio").decode()
    assert speech_payload["ref_text"] == "参考台词。"
    assert "voice" not in speech_payload
    await session.close()


async def test_tts_can_use_a_dedicated_local_endpoint(tmp_path) -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.host == "omlx.test" and request.url.path.endswith(
            "/chat/completions"
        ):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "你回来啦。我等你好久了。"}}]},
            )
        if request.url.host == "higgs.test" and request.url.path.endswith(
            "/audio/speech"
        ):
            return httpx.Response(
                200,
                content=_pcm16_wav(b"\x00\x00" * 2400, 24_000),
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    settings = Settings(
        _env_file=None,
        VH_VOICE_PROFILE_LOCKED=False,
        VH_UPSTREAM_MODE="omlx",
        VH_CHAT_BACKEND="omlx",
        VH_OMLX_API_KEY="local-secret",
        VH_OMLX_BASE_URL="http://omlx.test/v1",
        VH_TTS_BASE_URL="http://higgs.test/v1",
        VH_OMLX_TTS_MODEL="Higgs-TTS-3-4B-bf16",
        VH_TTS_FLOW_MODE="g_continuous",
    )
    session = OMLXRealtimeSession(settings, transport=httpx.MockTransport(handler))
    text = await session._chat("你在吗？")
    chunks = [chunk async for chunk in session._synthesize_stream(text)]

    assert chunks
    assert {request.url.host for request in observed} == {"omlx.test", "higgs.test"}
    speech_requests = [
        json.loads(request.content)
        for request in observed
        if request.url.path.endswith("/audio/speech")
    ]
    assert [request["input"] for request in speech_requests] == [
        "你回来啦。",
        "我等你好久了。",
    ]
    await session.close()


async def test_chat_can_use_ark_responses_while_asr_and_tts_stay_local() -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.host == "ark.test" and request.url.path.endswith("/responses"):
            return httpx.Response(
                200,
                json={
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "这需求改得也太突然了，先缓口气再处理。",
                                }
                            ],
                        }
                    ]
                },
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    settings = Settings(
        _env_file=None,
        VH_VOICE_PROFILE_LOCKED=False,
        VH_UPSTREAM_MODE="omlx",
        VH_CHAT_BACKEND="ark",
        VH_ARK_API_KEY="cloud-secret",
        VH_ARK_BASE_URL="https://ark.test/api/plan/v3",
        VH_ARK_MODEL="doubao-seed-2-1-turbo-260628",
        VH_OMLX_BASE_URL="http://omlx.test/v1",
        VH_TTS_BASE_URL="http://higgs.test/v1",
    )
    session = OMLXRealtimeSession(settings, transport=httpx.MockTransport(handler))
    text = await session._chat("老板又临时改需求。")

    assert text == "这需求改得也太突然了，先缓口气再处理。"
    assert len(observed) == 1
    request = observed[0]
    assert request.headers["authorization"] == "Bearer cloud-secret"
    payload = json.loads(request.content)
    assert payload["model"] == "doubao-seed-2-1-turbo-260628"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["store"] is False
    assert payload["input"][-1] == {
        "role": "user",
        "content": "老板又临时改需求。",
    }
    await session.close()


async def test_benchmark_session_can_disable_memory_recall_and_capture() -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "这次只按当前的话回答。"}}]},
            )
        raise AssertionError(f"Memory-isolated turn made an unexpected request: {request.url}")

    settings = Settings(
        _env_file=None,
        VH_VOICE_PROFILE_LOCKED=False,
        VH_MEMORY_ENABLED=True,
        VH_MEMORY_BASE_URL="http://memory.test",
        VH_UPSTREAM_MODE="omlx",
        VH_CHAT_BACKEND="omlx",
        VH_OMLX_BASE_URL="http://omlx.test/v1",
    )
    session = OMLXRealtimeSession(settings, transport=httpx.MockTransport(handler))
    await session.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {"memory": {"enabled": False}},
            }
        )
    )
    assert await session._chat("这是自动测试。") == "这次只按当前的话回答。"

    assert len(observed) == 1
    assert observed[0].url.path.endswith("/chat/completions")
    await session.close()


async def test_memory_fact_is_injected_and_committed_after_each_turn() -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.path.endswith("/search/recall"):
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "result": {
                        "entries": [
                            {
                                "type": "entities",
                                "uri": "viking://user/default/memories/entities/用户/小猫.md",
                                "content": "用户明确告知昵称是小猫。",
                            },
                            {
                                "type": "events",
                                "content": "助手过去误以为用户没有说过昵称。",
                            },
                        ]
                    },
                },
            )
        if request.url.path.endswith("/chat/completions"):
            payload = json.loads(request.content)
            system_prompt = payload["messages"][0]["content"]
            assert "[用户资料·最高优先] 用户明确告知昵称是小猫。" in system_prompt
            assert "优先于“历史事件”中助手自己说过的旧回答" in system_prompt
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "你叫小猫。"}}]},
            )
        if request.url.path.endswith("/messages/batch"):
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path.endswith("/commit"):
            assert json.loads(request.content) == {"keep_recent_count": 0}
            return httpx.Response(200, json={"status": "ok"})
        raise AssertionError(f"Unexpected request: {request.url}")

    settings = Settings(
        _env_file=None,
        VH_VOICE_PROFILE_LOCKED=False,
        VH_MEMORY_ENABLED=True,
        VH_MEMORY_BASE_URL="http://memory.test",
        VH_MEMORY_COMMIT_EVERY_TURNS=1,
        VH_UPSTREAM_MODE="omlx",
        VH_CHAT_BACKEND="omlx",
        VH_OMLX_BASE_URL="http://omlx.test/v1",
    )
    session = OMLXRealtimeSession(settings, transport=httpx.MockTransport(handler))

    assert await session._chat("我叫什么名字？") == "你叫小猫。"
    assert [request.url.path for request in observed] == [
        "/api/v1/search/recall",
        "/v1/chat/completions",
        f"/api/v1/sessions/{session.session_id}/messages/batch",
        f"/api/v1/sessions/{session.session_id}/commit",
    ]
    await session.close()


async def test_sglang_higgs_uses_locked_server_local_reference(tmp_path) -> None:
    observed: list[httpx.Request] = []
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"reference-audio")

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            content=_pcm16_wav(b"\x00\x00" * 2400, 24_000),
            headers={"content-type": "audio/wav"},
        )

    settings = Settings(
        _env_file=None,
        VH_VOICE_PROFILE_LOCKED=False,
        VH_UPSTREAM_MODE="omlx",
        VH_OMLX_BASE_URL="http://llm.test/v1",
        VH_TTS_BASE_URL="http://tts.test/v1",
        VH_TTS_PROTOCOL="sglang_higgs",
        VH_OMLX_TTS_REF_AUDIO=str(reference_audio),
        VH_OMLX_TTS_REF_TEXT="参考台词。",
    )
    session = OMLXRealtimeSession(settings, transport=httpx.MockTransport(handler))

    chunks = [chunk async for chunk in session._synthesize_stream("你回来啦。")]

    assert chunks
    request = observed[0]
    assert request.url.host == "tts.test"
    payload = json.loads(request.content)
    assert payload["stream"] is False
    assert payload["references"] == [
        {
            "audio_path": str(reference_audio.resolve()),
            "text": "参考台词。",
        }
    ]
    assert "model" not in payload
    assert "ref_audio" not in payload
    await session.close()


async def test_vllm_omni_higgs_uses_data_url_reference(tmp_path) -> None:
    observed: list[httpx.Request] = []
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"reference-audio")

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            content=_pcm16_wav(b"\x00\x00" * 2400, 24_000),
            headers={"content-type": "audio/wav"},
        )

    settings = Settings(
        _env_file=None,
        VH_VOICE_PROFILE_LOCKED=False,
        VH_UPSTREAM_MODE="omlx",
        VH_OMLX_BASE_URL="http://llm.test/v1",
        VH_TTS_BASE_URL="http://tts.test/v1",
        VH_TTS_PROTOCOL="vllm_omni_higgs",
        VH_TTS_SERVED_MODEL="higgs_audio_v3",
        VH_OMLX_TTS_REF_AUDIO=str(reference_audio),
        VH_OMLX_TTS_REF_TEXT="参考台词。",
        VH_TTS_SEED=20260817,
    )
    session = OMLXRealtimeSession(settings, transport=httpx.MockTransport(handler))

    chunks = [chunk async for chunk in session._synthesize_stream("你回来啦。")]

    assert chunks
    payload = json.loads(observed[0].content)
    assert payload["model"] == "higgs_audio_v3"
    assert payload["stream"] is False
    assert payload["response_format"] == "wav"
    assert payload["seed"] == 20260817
    assert payload["ref_audio"] == (
        "data:audio/wav;base64,"
        + base64.b64encode(b"reference-audio").decode("ascii")
    )
    assert payload["ref_text"] == "参考台词。"
    assert "references" not in payload
    await session.close()


async def test_short_filler_noise_does_not_trigger_a_chat_reply() -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": "嗯。"})
        raise AssertionError(f"Noise must not reach chat or TTS: {request.url}")

    settings = Settings(
        VH_UPSTREAM_MODE="omlx",
        VH_OMLX_API_KEY="local-secret",
        VH_OMLX_BASE_URL="http://omlx.test/v1",
    )
    session = OMLXRealtimeSession(settings, transport=httpx.MockTransport(handler))
    await session.start()
    events = session.events()
    assert json.loads(await anext(events))["type"] == "session.created"

    pcm = b"\x00\x00" * 12_000  # 0.5 seconds at 24 kHz.
    await session.send(
        json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )
    )
    await session.send(json.dumps({"type": "input_audio_buffer.commit"}))

    event_types: list[str] = []
    while event_types[-1:] != ["input_audio_buffer.no_speech"]:
        event_types.append(json.loads(await anext(events))["type"])

    assert event_types == [
        "input_audio_buffer.speech_started",
        "conversation.item.input_audio_transcription.completed",
        "input_audio_buffer.no_speech",
    ]
    assert len(observed) == 1
    assert observed[0].url.path.endswith("/audio/transcriptions")
    await session.close()


async def test_asr_can_use_a_separate_local_service() -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(200, json={"text": "单独的本地语音识别。"})

    settings = Settings(
        _env_file=None,
        VH_VOICE_PROFILE_LOCKED=False,
        VH_UPSTREAM_MODE="omlx",
        VH_OMLX_BASE_URL="http://llm.test/v1",
        VH_ASR_BASE_URL="http://asr.test/v1",
    )
    session = OMLXRealtimeSession(settings, transport=httpx.MockTransport(handler))

    assert await session._transcribe(b"\x00\x00" * 2400) == "单独的本地语音识别。"
    assert observed[0].url.host == "asr.test"
    await session.close()
