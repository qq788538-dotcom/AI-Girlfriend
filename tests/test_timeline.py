import pytest

from virtual_human.timeline import AudioTimeline


def test_audio_timeline_uses_audio_as_master_clock() -> None:
    timeline = AudioTimeline(sample_rate=24000)

    first = timeline.ingest("resp_1", b"\x00\x00" * 960)
    second = timeline.ingest("resp_1", b"\x00\x00" * 480)

    assert first.chunk_index == 0
    assert first.pts_ms == 0
    assert first.duration_ms == pytest.approx(40)
    assert second.chunk_index == 1
    assert second.pts_ms == pytest.approx(40)
    assert second.duration_ms == pytest.approx(20)
    assert timeline.buffered_duration_ms == pytest.approx(60)


def test_new_response_starts_new_timeline() -> None:
    timeline = AudioTimeline(sample_rate=16000)
    timeline.ingest("resp_old", b"\x00\x00" * 1600)

    chunk = timeline.ingest("resp_new", b"\x00\x00" * 320)

    assert chunk.chunk_index == 0
    assert chunk.pts_ms == 0
    assert chunk.duration_ms == pytest.approx(20)


def test_rejects_partial_pcm_sample() -> None:
    timeline = AudioTimeline(sample_rate=24000)

    with pytest.raises(ValueError, match="whole number of samples"):
        timeline.ingest("resp_1", b"\x00")
