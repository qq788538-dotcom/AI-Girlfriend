import base64
import wave

import pytest

from virtual_human.renderer_worker import LiveActOfficialBackend, ResponseAudio


def _audio_event(chunk_index: int, sample_count: int = 960) -> dict[str, object]:
    pcm = b"\x00\x00" * sample_count
    return {
        "chunk_index": chunk_index,
        "sample_rate": 24_000,
        "audio": base64.b64encode(pcm).decode("ascii"),
    }


def test_response_audio_enforces_order_and_writes_wav(tmp_path) -> None:
    response = ResponseAudio(response_id="response-1", sample_rate=24_000)
    response.append(_audio_event(0))
    response.append(_audio_event(1))

    assert response.duration_ms == 80

    with pytest.raises(ValueError, match="Expected chunk 2"):
        response.append(_audio_event(3))

    wav_path = tmp_path / "response.wav"
    response.write_wav(wav_path)
    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getframerate() == 24_000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 1_920


async def test_liveact_publishes_final_mp4_for_browser_playback(
    tmp_path, monkeypatch
) -> None:
    runtime_dir = tmp_path / "runtime"
    output_dir = runtime_dir / "sessions" / "session-1"
    output_dir.mkdir(parents=True)
    reference = output_dir / "reference.jpg"
    reference.write_bytes(b"image")
    final_video = tmp_path / "liveact-final.mp4"
    final_video.write_bytes(b"mp4")
    response = ResponseAudio(response_id="response-1", sample_rate=24_000)
    response.append(_audio_event(0))
    events: list[dict[str, object]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "is_done": True,
                "stream_ready": False,
                "final_video_path": str(final_video),
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, *_args, **_kwargs) -> FakeResponse:
            return FakeResponse()

        async def get(self, *_args, **_kwargs) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(
        "virtual_human.renderer_worker.httpx.AsyncClient",
        lambda **_kwargs: FakeClient(),
    )
    backend = LiveActOfficialBackend("http://127.0.0.1:5001", 24, "talk")

    await backend.render(
        session_id="session-1",
        response=response,
        reference_path=reference,
        output_dir=output_dir,
        base_url="http://127.0.0.1:8772",
        emit=lambda event: _capture_event(events, event),
    )

    published = output_dir / "response-1.mp4"
    assert published.read_bytes() == b"mp4"
    assert events[-2] == {
        "type": "avatar.video.ready",
        "session_id": "session-1",
        "response_id": "response-1",
        "url": "http://127.0.0.1:8772/runtime/sessions/session-1/response-1.mp4",
        "duration_ms": 40,
        "backend": "liveact-official",
        "clock": "audio-master",
        "audio_included": True,
    }
    assert events[-1]["type"] == "avatar.render.done"
    assert all(event["type"] != "avatar.stream.ready" for event in events)


async def _capture_event(
    events: list[dict[str, object]], event: dict[str, object]
) -> None:
    events.append(event)
