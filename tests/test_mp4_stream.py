import shutil

import pytest

from virtual_human.av_verify import probe_media
from virtual_human.mp4_stream import (
    GENTLE_UNSHARP_FILTER,
    ContinuousMP4Muxer,
    MP4BoxParser,
    MP4FragmentAssembler,
    build_video_filter,
)


def _box(box_type: bytes, payload: bytes = b"") -> bytes:
    return (8 + len(payload)).to_bytes(4, "big") + box_type + payload


def test_box_parser_handles_partial_network_chunks() -> None:
    parser = MP4BoxParser()
    box = _box(b"ftyp", b"abcd")

    assert parser.feed(box[:5]) == []
    assert parser.feed(box[5:]) == [(b"ftyp", box)]
    parser.finish()


def test_fragment_assembler_separates_init_and_media_fragments() -> None:
    init = _box(b"ftyp") + _box(b"moov")
    first = _box(b"moof") + _box(b"mdat", b"one")
    second = _box(b"moof") + _box(b"mdat", b"two")
    assembler = MP4FragmentAssembler()

    outputs = assembler.feed(init + first + second)

    assert outputs == [
        ("init", init),
        ("media", first),
        ("media", second),
    ]
    assembler.finish()


def test_video_filter_upscales_before_gentle_sharpening() -> None:
    assert (
        build_video_filter(
            input_width=512,
            input_height=512,
            output_width=768,
            output_height=768,
            sharpen=True,
        )
        == f"scale=768:768:flags=lanczos,{GENTLE_UNSHARP_FILTER}"
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
async def test_continuous_muxer_emits_playable_fragmented_mp4(tmp_path) -> None:
    outputs: list[tuple[str, bytes]] = []

    async def emit(kind: str, payload: bytes) -> None:
        outputs.append((kind, payload))

    muxer = ContinuousMP4Muxer(
        ffmpeg_bin="ffmpeg",
        width=64,
        height=64,
        fps=10,
        sample_rate=16_000,
        gop_size=10,
        emit=emit,
        output_width=96,
        output_height=96,
    )
    assert muxer.sharpen is True
    assert GENTLE_UNSHARP_FILTER == "unsharp=5:5:0.55:3:3:0.15"
    rgb_frame = bytes([24, 32, 48]) * 64 * 64
    await muxer.write(rgb_frame * 10, b"\x00\x00" * 16_000)
    await muxer.write(rgb_frame * 10, b"\x00\x00" * 16_000)
    await muxer.finish()

    assert outputs[0][0] == "init"
    assert [kind for kind, _ in outputs].count("media") >= 1

    media_path = tmp_path / "stream.mp4"
    media_path.write_bytes(b"".join(payload for _, payload in outputs))
    probe = probe_media(str(media_path))
    assert {stream["codec_type"] for stream in probe["streams"]} == {"audio", "video"}
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    assert (video["width"], video["height"]) == (96, 96)
