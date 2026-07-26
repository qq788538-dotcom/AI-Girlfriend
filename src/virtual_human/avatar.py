from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any

import websockets

from virtual_human.timeline import AudioChunk

logger = logging.getLogger(__name__)

RENDERER_PING_TIMEOUT_SECONDS = 180


class AvatarSink(ABC):
    @abstractmethod
    async def start(self, session_id: str, *, reference: str, sample_rate: int) -> None: ...

    @abstractmethod
    async def set_state(self, state: str, *, response_id: str | None = None) -> None: ...

    @abstractmethod
    async def push_audio(self, chunk: AudioChunk) -> None: ...

    @abstractmethod
    async def finish(self, response_id: str | None) -> None: ...

    @abstractmethod
    async def cancel(self, response_id: str | None) -> None: ...

    @abstractmethod
    async def events(self): ...

    @abstractmethod
    async def close(self) -> None: ...


class MockAvatarSink(AvatarSink):
    """A no-GPU sink used to verify timing and cancellation locally."""

    def __init__(self) -> None:
        self.session_id: str | None = None
        self.state = "idle"
        self.chunks: list[AudioChunk] = []

    async def start(self, session_id: str, *, reference: str, sample_rate: int) -> None:
        self.session_id = session_id
        logger.info("Mock avatar started: session=%s reference=%s rate=%d", session_id, reference, sample_rate)

    async def set_state(self, state: str, *, response_id: str | None = None) -> None:
        self.state = state

    async def push_audio(self, chunk: AudioChunk) -> None:
        self.chunks.append(chunk)

    async def finish(self, response_id: str | None) -> None:
        self.state = "listening"

    async def cancel(self, response_id: str | None) -> None:
        self.chunks = [chunk for chunk in self.chunks if chunk.response_id != response_id]
        self.state = "listening"

    async def events(self):
        if False:
            yield {}

    async def close(self) -> None:
        self.state = "closed"


class WebSocketAvatarSink(AvatarSink):
    """Protocol adapter for a remote SoulX-LiveAct/FlashHead rendering worker.

    Messages are JSON so the GPU worker can be implemented in Python without
    coupling it to this gateway. PCM is base64 encoded during the first
    integration phase; a binary transport can replace it after profiling.
    """

    def __init__(self, url: str, backend: str, preroll_ms: int, token: str = "") -> None:
        self.url = url
        self.backend = backend
        self.preroll_ms = preroll_ms
        self.token = token
        self._ws: Any | None = None
        self._lock = asyncio.Lock()

    async def _send(self, payload: dict[str, Any]) -> bool:
        if self._ws is None:
            return False
        async with self._lock:
            try:
                await self._ws.send(json.dumps(payload, ensure_ascii=False))
            except websockets.ConnectionClosed:
                self._ws = None
                return False
        return True

    async def start(self, session_id: str, *, reference: str, sample_rate: int) -> None:
        additional_headers = (
            {"Authorization": f"Bearer {self.token}"}
            if self.token
            else None
        )
        self._ws = await websockets.connect(
            self.url,
            max_size=None,
            ping_interval=20,
            ping_timeout=RENDERER_PING_TIMEOUT_SECONDS,
            additional_headers=additional_headers,
        )
        reference_payload: dict[str, Any] = {"reference": reference}
        reference_path = Path(reference)
        if reference_path.is_file():
            reference_payload.update(
                {
                    "reference_name": reference_path.name,
                    "reference_mime": mimetypes.guess_type(reference_path.name)[0] or "image/jpeg",
                    "reference_image": base64.b64encode(reference_path.read_bytes()).decode("ascii"),
                }
            )
        await self._send(
            {
                "type": "avatar.session.start",
                "session_id": session_id,
                "backend": self.backend,
                **reference_payload,
                "sample_rate": sample_rate,
                "preroll_ms": self.preroll_ms,
                "clock": "audio-master",
            }
        )

    async def set_state(self, state: str, *, response_id: str | None = None) -> None:
        await self._send({"type": "avatar.state", "state": state, "response_id": response_id})

    async def push_audio(self, chunk: AudioChunk) -> None:
        metadata = asdict(chunk)
        metadata.pop("pcm16")
        await self._send(
            {
                "type": "avatar.audio.append",
                **metadata,
                "audio": base64.b64encode(chunk.pcm16).decode("ascii"),
            }
        )

    async def finish(self, response_id: str | None) -> None:
        await self._send({"type": "avatar.response.finish", "response_id": response_id})

    async def cancel(self, response_id: str | None) -> None:
        await self._send({"type": "avatar.response.cancel", "response_id": response_id})

    async def events(self):
        if self._ws is None:
            return
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Ignoring invalid renderer event")
                    continue
                if isinstance(event, dict):
                    yield event
        except websockets.ConnectionClosed as error:
            logger.warning("Avatar renderer disconnected: %s", error)
            self._ws = None
            yield {
                "type": "avatar.error",
                "message": "人物渲染节点已断开，当前回答改用纯语音。",
                "recoverable": True,
            }

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


def build_avatar_sink(
    url: str,
    backend: str,
    preroll_ms: int,
    token: str = "",
) -> AvatarSink:
    if url:
        return WebSocketAvatarSink(url, backend, preroll_ms, token)
    return MockAvatarSink()
