from __future__ import annotations

import io
import wave

from virtual_human.higgs_server import _streaming_wav_header, _wav_bytes
from virtual_human.omlx_realtime import _parse_streaming_wav_header


def test_higgs_server_builds_24khz_mono_pcm16_wav() -> None:
    payload = _wav_bytes(b"\x01\x00\xff\xff", 24_000)

    with wave.open(io.BytesIO(payload), "rb") as wav_file:
        assert wav_file.getframerate() == 24_000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.readframes(2) == b"\x01\x00\xff\xff"


def test_higgs_server_builds_parseable_streaming_wav_header() -> None:
    header = _streaming_wav_header(24_000)

    assert len(header) == 44
    assert _parse_streaming_wav_header(header) == (44, 24_000, 1, 2)
    assert int.from_bytes(header[4:8], "little") == 0xFFFFFFFF
    assert int.from_bytes(header[40:44], "little") == 0xFFFFFFFF
