#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

from virtual_human.config import Settings
from virtual_human.omlx_realtime import OMLXRealtimeSession


async def benchmark(text: str, output: Path, temperature: float | None) -> dict[str, object]:
    settings = (
        Settings(VH_OMLX_TTS_TEMPERATURE=temperature)
        if temperature is not None
        else Settings()
    )
    session = OMLXRealtimeSession(settings)
    pcm = bytearray()
    started = time.perf_counter()
    first_chunk_ms: float | None = None
    try:
        async for chunk in session._synthesize_stream(text):
            if first_chunk_ms is None:
                first_chunk_ms = (time.perf_counter() - started) * 1000
            pcm.extend(chunk)
    finally:
        await session.close()

    total_ms = (time.perf_counter() - started) * 1000
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(settings.output_sample_rate)
        wav_file.writeframes(pcm)

    audio_duration_s = len(pcm) / (settings.output_sample_rate * 2)
    return {
        "model": settings.omlx_tts_model,
        "reference_conditioned": bool(settings.omlx_tts_ref_audio),
        "temperature": settings.omlx_tts_temperature,
        "top_p": settings.omlx_tts_top_p,
        "repetition_penalty": settings.omlx_tts_repetition_penalty,
        "text": text,
        "first_pcm_chunk_ms": round(first_chunk_ms or 0.0, 1),
        "total_generation_ms": round(total_ms, 1),
        "audio_duration_s": round(audio_duration_s, 3),
        "realtime_factor": round(total_ms / 1000 / audio_duration_s, 3),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local reference-conditioned TTS.")
    parser.add_argument(
        "--text",
        default="你回来啦？我刚刚还有一点想你，不过才不会轻易承认呢。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runtime/voice-calibration/corrected-realtime-sample.wav"),
    )
    parser.add_argument("--temperature", type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(benchmark(args.text, args.output, args.temperature)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
