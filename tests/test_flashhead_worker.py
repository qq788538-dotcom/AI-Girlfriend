import base64
from collections import deque

import pytest

from virtual_human.flashhead_worker import FlashHeadEngine, FlashHeadSettings, StreamingResponse


def _append_event(chunk_index: int, samples: int = 960) -> dict[str, object]:
    return {
        "chunk_index": chunk_index,
        "sample_rate": 24_000,
        "audio": base64.b64encode(b"\x00\x00" * samples).decode("ascii"),
    }


def test_streaming_response_validates_monotonic_pcm_chunks() -> None:
    response = StreamingResponse("response-1", 24_000)
    response.append(_append_event(0))

    assert response.expected_chunk_index == 1
    assert len(response.pcm16) == 1_920

    with pytest.raises(ValueError, match="Expected chunk 1"):
        response.append(_append_event(2))


def test_flashhead_lite_uses_about_one_second_input_slices() -> None:
    engine = FlashHeadEngine(FlashHeadSettings(VH_FLASHHEAD_REFERENCE_LOCKED=False))
    engine.params = {
        "tgt_fps": 25,
        "frame_num": 33,
        "motion_frames_num": 9,
    }

    assert engine.slice_frames == 24
    assert engine.input_slice_bytes(24_000) == 23_040 * 2


def test_flashhead_audio_motion_scale_is_opt_in() -> None:
    default_engine = FlashHeadEngine(
        FlashHeadSettings(VH_FLASHHEAD_REFERENCE_LOCKED=False)
    )
    damped_engine = FlashHeadEngine(
        FlashHeadSettings(
            VH_FLASHHEAD_REFERENCE_LOCKED=False,
            VH_FLASHHEAD_AUDIO_MOTION_SCALE=0.65,
        )
    )

    assert default_engine.scale_audio_embedding(10.0) == 10.0
    assert damped_engine.scale_audio_embedding(10.0) == pytest.approx(6.5)


def test_flashhead_audio_motion_scale_rejects_unsafe_values() -> None:
    with pytest.raises(ValueError):
        FlashHeadSettings(
            VH_FLASHHEAD_REFERENCE_LOCKED=False,
            VH_FLASHHEAD_AUDIO_MOTION_SCALE=0.1,
        )


def test_flashhead_access_token_is_read_from_a_separate_file(tmp_path) -> None:
    token_file = tmp_path / "renderer.token"
    token_file.write_text("private-renderer-token\n")
    settings = FlashHeadSettings(
        VH_FLASHHEAD_REFERENCE_LOCKED=False,
        VH_FLASHHEAD_ACCESS_TOKEN_FILE=str(token_file),
    )

    assert settings.resolved_access_token() == "private-renderer-token"


def test_flashhead_reuses_identical_character_reference(tmp_path) -> None:
    class Pipeline:
        def __init__(self) -> None:
            self.reset_count = 0

        def reset_person_name(self) -> None:
            self.reset_count += 1

    engine = FlashHeadEngine(FlashHeadSettings(VH_FLASHHEAD_REFERENCE_LOCKED=False))
    engine.pipeline = Pipeline()
    engine.params = {"sample_rate": 16_000, "cached_audio_duration": 2}
    engine.audio_history = deque(maxlen=32_000)
    prepared: list[str] = []
    engine._get_base_data = lambda *_args, **_kwargs: prepared.append("prepared")
    engine.load = lambda: None

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"same-reference")
    second.write_bytes(b"same-reference")

    engine.prepare_character(first)
    engine.prepare_character(second)

    assert prepared == ["prepared"]
    assert engine.pipeline.reset_count == 1


def test_flashhead_rejects_unapproved_character_reference(tmp_path) -> None:
    engine = FlashHeadEngine(FlashHeadSettings())
    engine.load = lambda: None
    reference = tmp_path / "other-person.png"
    reference.write_bytes(b"unapproved-character")

    with pytest.raises(ValueError, match="identity is locked"):
        engine.prepare_character(reference)
