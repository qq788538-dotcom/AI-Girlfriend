from __future__ import annotations

from fastapi.testclient import TestClient

from virtual_human.asr_server import create_asr_app


class FakeASRBackend:
    model_name = "local-qwen-asr"

    def transcribe(self, audio: bytes, language: str | None = None) -> str:
        assert audio == b"RIFF-local-test"
        assert language == "Chinese"
        return "我回来了。"


def test_asr_server_exposes_openai_transcription_endpoint() -> None:
    client = TestClient(create_asr_app(FakeASRBackend()))

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("input.wav", b"RIFF-local-test", "audio/wav")},
        data={"model": "Qwen3-ASR-0.6B", "language": "Chinese"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "我回来了。"}
    assert client.get("/healthz").json() == {
        "status": "ok",
        "model": "local-qwen-asr",
        "device": "cuda:0",
        "local_files_only": True,
    }


def test_asr_server_rejects_empty_audio() -> None:
    client = TestClient(create_asr_app(FakeASRBackend()))

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("input.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400
