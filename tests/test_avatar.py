from websockets.exceptions import ConnectionClosedError

from virtual_human.avatar import MockAvatarSink, WebSocketAvatarSink
from virtual_human.timeline import AudioChunk


async def test_mock_avatar_cancels_only_target_response() -> None:
    sink = MockAvatarSink()
    await sink.start("session", reference="avatar.jpg", sample_rate=24000)
    first = AudioChunk("resp_1", 0, 0, 20, 24000, b"\x00\x00" * 480)
    second = AudioChunk("resp_2", 0, 0, 20, 24000, b"\x00\x00" * 480)
    await sink.push_audio(first)
    await sink.push_audio(second)

    await sink.cancel("resp_1")

    assert sink.chunks == [second]
    assert sink.state == "listening"


async def test_mock_avatar_finish_returns_to_listening() -> None:
    sink = MockAvatarSink()
    await sink.start("session", reference="avatar.jpg", sample_rate=24000)
    await sink.set_state("speaking", response_id="resp_1")

    await sink.finish("resp_1")

    assert sink.state == "listening"


class _DisconnectedRenderer:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise ConnectionClosedError(None, None)

    async def send(self, payload: str) -> None:
        raise ConnectionClosedError(None, None)


async def test_remote_avatar_degrades_when_renderer_disconnects() -> None:
    sink = WebSocketAvatarSink("ws://renderer/avatar", "flashhead-lite", 1000)
    sink._ws = _DisconnectedRenderer()

    event = await anext(sink.events())

    assert event["type"] == "avatar.error"
    assert event["recoverable"] is True
    assert sink._ws is None
    assert await sink._send({"type": "avatar.state"}) is False
