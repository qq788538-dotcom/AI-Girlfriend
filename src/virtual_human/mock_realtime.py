from __future__ import annotations

import asyncio
import base64
import json
import math
import struct
from collections.abc import AsyncIterator
from uuid import uuid4


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:18]}"


def _tone(sample_rate: int, seconds: float = 2.4) -> bytes:
    samples = bytearray()
    for index in range(int(sample_rate * seconds)):
        envelope = min(1.0, index / (sample_rate * 0.08), (sample_rate * seconds - index) / (sample_rate * 0.12))
        carrier = math.sin(2 * math.pi * 220 * index / sample_rate)
        harmonic = 0.35 * math.sin(2 * math.pi * 330 * index / sample_rate)
        value = int(max(-1, min(1, (carrier + harmonic) * 0.18 * envelope)) * 32767)
        samples.extend(struct.pack("<h", value))
    return bytes(samples)


class MockRealtimeSession:
    """Deterministic upstream used before an API key is supplied."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.session_id = _id("sess")
        self._events: asyncio.Queue[dict | None] = asyncio.Queue()
        self._response_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self._events.put(
            {
                "type": "session.created",
                "event_id": _id("event"),
                "session": {
                    "id": self.session_id,
                    "type": "realtime",
                    "model": "virtual-human-mock",
                    "audio": {
                        "input": {"format": {"type": "audio/pcm", "rate": self.sample_rate}},
                        "output": {"format": {"type": "audio/pcm", "rate": self.sample_rate}, "voice": "mock"},
                    },
                },
            }
        )

    async def send(self, raw: str) -> None:
        event = json.loads(raw)
        event_type = event.get("type")
        if event_type == "session.update":
            await self._events.put(
                {
                    "type": "session.updated",
                    "event_id": _id("event"),
                    "session": event.get("session", {}),
                }
            )
        elif event_type == "response.create":
            if self._response_task and not self._response_task.done():
                return
            self._response_task = asyncio.create_task(self._stream_response())
        elif event_type == "response.cancel":
            await self.cancel()

    async def _stream_response(self) -> None:
        response_id = _id("resp")
        item_id = _id("item")
        await self._events.put(
            {
                "type": "response.created",
                "event_id": _id("event"),
                "response": {"id": response_id, "status": "in_progress", "output": []},
            }
        )
        pcm = _tone(self.sample_rate)
        chunk_bytes = int(self.sample_rate * 0.04) * 2
        for offset in range(0, len(pcm), chunk_bytes):
            await self._events.put(
                {
                    "type": "response.output_audio.delta",
                    "event_id": _id("event"),
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": base64.b64encode(pcm[offset : offset + chunk_bytes]).decode("ascii"),
                }
            )
            await asyncio.sleep(0.02)
        transcript = "这是本机模拟链路。接入 OpenAI API 后，我会使用真实自然语音和情绪与你对话。"
        await self._events.put(
            {
                "type": "response.output_audio_transcript.done",
                "event_id": _id("event"),
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "transcript": transcript,
            }
        )
        await self._events.put(
            {
                "type": "response.output_audio.done",
                "event_id": _id("event"),
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
            }
        )
        await self._events.put(
            {
                "type": "response.done",
                "event_id": _id("event"),
                "response": {"id": response_id, "status": "completed", "output": []},
            }
        )

    async def cancel(self) -> None:
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass
        self._response_task = None

    async def events(self) -> AsyncIterator[str]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield json.dumps(event)

    async def close(self) -> None:
        await self.cancel()
        await self._events.put(None)
