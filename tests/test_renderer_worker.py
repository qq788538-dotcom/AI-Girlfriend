import base64
import wave

import pytest

from virtual_human.renderer_worker import ResponseAudio


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
