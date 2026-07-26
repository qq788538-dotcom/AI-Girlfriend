#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import wave
from pathlib import Path

from virtual_human.config import Settings
from virtual_human.omlx_realtime import OMLXRealtimeSession


async def transcribe(path: Path) -> str:
    with wave.open(str(path), "rb") as wav_file:
        if (
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getframerate(),
        ) != (1, 2, 24_000):
            raise ValueError(f"{path} must be 24 kHz mono PCM16 WAV")
        pcm16 = wav_file.readframes(wav_file.getnframes())

    session = OMLXRealtimeSession(Settings())
    try:
        return await session._transcribe(pcm16)
    finally:
        await session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a local WAV through oMLX.")
    parser.add_argument("audio", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.audio:
        print(f"{path}: {asyncio.run(transcribe(path))}")


if __name__ == "__main__":
    main()
