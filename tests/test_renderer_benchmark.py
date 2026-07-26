from __future__ import annotations

import wave

import pytest
import websockets

from virtual_human.renderer_benchmark import (
    BenchmarkMetrics,
    benchmark_renderer,
    benchmark_violations,
)


def _metrics() -> BenchmarkMetrics:
    return BenchmarkMetrics(
        renderer_url="ws://renderer/avatar",
        session_ready_ms=800,
        audio_duration_ms=10_000,
        audio_feed_ms=10_000,
        first_output_ms=1_150,
        render_done_ms=10_700,
        post_audio_tail_ms=700,
        end_to_end_realtime_factor=0.935,
        realtime_input=True,
        audio_chunks=250,
        media_fragments=10,
        fallback_segments=0,
        backend="flashhead-lite",
        output_path="/tmp/output.mp4",
    )


def test_benchmark_thresholds_accept_realtime_renderer() -> None:
    assert (
        benchmark_violations(
            _metrics(),
            max_first_output_ms=1_500,
            max_post_audio_tail_ms=1_000,
            min_realtime_factor=0.9,
        )
        == []
    )


def test_benchmark_thresholds_report_each_failure() -> None:
    violations = benchmark_violations(
        _metrics(),
        max_first_output_ms=1_000,
        max_post_audio_tail_ms=500,
        min_realtime_factor=1.0,
    )

    assert len(violations) == 3
    assert "first output" in violations[0]
    assert "post-audio tail" in violations[1]
    assert "realtime factor" in violations[2]


async def test_benchmark_reports_connection_closed_before_ready(tmp_path) -> None:
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"not-a-real-image")
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * 160)

    async def close_before_ready(connection) -> None:
        await connection.recv()
        await connection.close(code=1011, reason="model load failed")

    async with websockets.serve(close_before_ready, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        with pytest.raises(RuntimeError, match="connection closed before completion"):
            await benchmark_renderer(
                renderer_url=f"ws://127.0.0.1:{port}",
                reference_path=reference,
                audio_path=audio,
                output_path=None,
                chunk_ms=40,
                realtime_input=False,
                timeout_seconds=2,
            )
