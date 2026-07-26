import asyncio
import base64
import json

import websockets

from virtual_human.avatar import MockAvatarSink
from virtual_human.bridge import RealtimeBridge
from virtual_human.config import Settings


class FakeBrowserWebSocket:
    def __init__(self, events: list[dict]) -> None:
        self.incoming: asyncio.Queue[str] = asyncio.Queue()
        for event in events:
            self.incoming.put_nowait(json.dumps(event))
        self.sent_text: list[dict] = []
        self.sent_json: list[dict] = []

    async def receive_text(self) -> str:
        return await self.incoming.get()

    async def send_text(self, raw: str) -> None:
        self.sent_text.append(json.loads(raw))

    async def send_json(self, event: dict) -> None:
        self.sent_json.append(event)


class UnavailableAvatarSink(MockAvatarSink):
    async def start(self, session_id: str, *, reference: str, sample_rate: int) -> None:
        raise ConnectionRefusedError("cloud renderer is offline")


async def test_bridge_falls_back_when_cloud_avatar_is_offline() -> None:
    settings = Settings(
        _env_file=None,
        VH_AVATAR_REFERENCE="public/avatar-reference.jpg",
        VH_AVATAR_REFERENCE_LOCKED=False,
    )
    browser = FakeBrowserWebSocket([])
    bridge = RealtimeBridge(settings, browser, UnavailableAvatarSink())

    await bridge._start_avatar()

    assert isinstance(bridge.avatar, MockAvatarSink)
    assert browser.sent_json == [
        {
            "type": "avatar.fallback",
            "playback_owner": "browser",
            "message": "云端人物动画未连接，已切换为本地静态人物和浏览器语音。",
            "recoverable": True,
        }
    ]


async def test_mock_avatar_announces_browser_playback_mode() -> None:
    settings = Settings(
        _env_file=None,
        VH_AVATAR_REFERENCE="public/avatar-reference.jpg",
        VH_AVATAR_REFERENCE_LOCKED=False,
    )
    browser = FakeBrowserWebSocket([])
    bridge = RealtimeBridge(settings, browser, MockAvatarSink())

    await bridge._start_avatar()

    assert browser.sent_json == [
        {"type": "avatar.mode", "playback_owner": "browser"}
    ]


async def test_openai_mode_forwards_auth_events_and_avatar_audio() -> None:
    observed: dict[str, object] = {}
    pcm = b"\x00\x00" * 480

    async def upstream(connection) -> None:
        observed["authorization"] = connection.request.headers["Authorization"]
        observed["path"] = connection.request.path
        observed["session_update"] = json.loads(await connection.recv())
        await connection.send(json.dumps({"type": "session.created", "session": {"id": "session-1"}}))
        observed["response_create"] = json.loads(await connection.recv())
        await connection.send(json.dumps({"type": "response.created", "response": {"id": "response-1"}}))
        await connection.send(
            json.dumps(
                {
                    "type": "response.output_audio.delta",
                    "response_id": "response-1",
                    "delta": base64.b64encode(pcm).decode("ascii"),
                }
            )
        )
        await connection.send(json.dumps({"type": "response.done", "response": {"id": "response-1"}}))

    async with websockets.serve(upstream, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        settings = Settings(
            VH_UPSTREAM_MODE="openai",
            OPENAI_API_KEY="test-secret",
            VH_OPENAI_REALTIME_URL=f"ws://127.0.0.1:{port}/v1/realtime",
            VH_OPENAI_REALTIME_MODEL="test-realtime",
            VH_AVATAR_REFERENCE="public/avatar-reference.jpg",
            VH_AVATAR_REFERENCE_LOCKED=False,
        )
        browser = FakeBrowserWebSocket(
            [
                {"type": "session.update", "session": {"output_modalities": ["audio"]}},
                {"type": "response.create", "response": {"output_modalities": ["audio"]}},
            ]
        )
        avatar = MockAvatarSink()
        bridge = RealtimeBridge(settings, browser, avatar)

        await bridge.run()

    assert observed["authorization"] == "Bearer test-secret"
    assert observed["path"] == "/v1/realtime?model=test-realtime"
    assert observed["session_update"]["type"] == "session.update"
    assert observed["response_create"]["type"] == "response.create"
    assert [event["type"] for event in browser.sent_text] == [
        "session.created",
        "response.created",
        "response.output_audio.delta",
        "response.done",
    ]
    assert len(avatar.chunks) == 1
    assert avatar.chunks[0].pcm16 == pcm
    assert any(event["type"] == "avatar.audio.timeline" for event in browser.sent_json)
