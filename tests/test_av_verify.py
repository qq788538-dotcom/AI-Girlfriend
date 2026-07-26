import pytest

from virtual_human.av_verify import validate_probe


def _probe(audio_duration: str = "2.400000") -> dict:
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 416,
                "height": 720,
                "r_frame_rate": "20/1",
                "duration": "2.400000",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "24000",
                "channels": 1,
                "duration": audio_duration,
            },
        ],
        "format": {"duration": "2.400000"},
    }


def test_validate_probe_accepts_synchronized_target_media() -> None:
    metrics = validate_probe(
        _probe(),
        expected_width=416,
        expected_height=720,
        expected_fps=20,
        expected_sample_rate=24000,
        max_drift_ms=80,
    )

    assert metrics.drift_ms == 0
    assert metrics.duration_ms == 2400


def test_validate_probe_rejects_excessive_drift() -> None:
    with pytest.raises(ValueError, match="duration drift"):
        validate_probe(
            _probe(audio_duration="2.100000"),
            expected_width=416,
            expected_height=720,
            expected_fps=20,
            expected_sample_rate=24000,
            max_drift_ms=80,
        )
