from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from typing import Any
from uuid import uuid4

import websockets
from fastapi import WebSocket, WebSocketDisconnect

from virtual_human.avatar import AvatarSink, MockAvatarSink
from virtual_human.config import Settings
from virtual_human.mock_realtime import MockRealtimeSession
from virtual_human.omlx_realtime import OMLXRealtimeSession
from virtual_human.timeline import AudioTimeline

logger = logging.getLogger(__name__)


class RealtimeBridge:
    """Bidirectional OpenAI-Realtime-compatible gateway with avatar audio tee."""

    def __init__(self, settings: Settings, client: WebSocket, avatar: AvatarSink) -> None:
        self.settings = settings
        self.client = client
        self.avatar = avatar
        self.timeline = AudioTimeline(settings.output_sample_rate)
        self.session_id = f"vh_{uuid4().hex[:20]}"
        self.current_response_id: str | None = None
        self._client_send_lock = asyncio.Lock()

    async def run(self) -> None:
        await self._start_avatar()
        avatar_events = asyncio.create_task(self._avatar_to_client())
        try:
            if self.settings.upstream_mode == "openai":
                await self._run_openai()
            elif self.settings.upstream_mode == "omlx":
                await self._run_omlx()
            else:
                await self._run_mock()
        finally:
            avatar_events.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await avatar_events
            await self.avatar.close()

    async def _start_avatar(self) -> None:
        try:
            await self.avatar.start(
                self.session_id,
                reference=self.settings.avatar_reference,
                sample_rate=self.settings.output_sample_rate,
            )
        except (ConnectionError, OSError, TimeoutError) as error:
            logger.warning(
                "Avatar renderer unavailable; using browser audio fallback: %s",
                error,
            )
            with contextlib.suppress(Exception):
                await self.avatar.close()
            self.avatar = MockAvatarSink()
            await self.avatar.start(
                self.session_id,
                reference=self.settings.avatar_reference,
                sample_rate=self.settings.output_sample_rate,
            )
            await self._send_client_json(
                {
                    "type": "avatar.fallback",
                    "playback_owner": "browser",
                    "message": "云端人物动画未连接，已切换为本地静态人物和浏览器语音。",
                    "recoverable": True,
                }
            )
            return
        await self._send_client_json(
            {
                "type": "avatar.mode",
                "playback_owner": (
                    "browser" if isinstance(self.avatar, MockAvatarSink) else "renderer"
                ),
            }
        )

    async def _run_openai(self) -> None:
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        async with websockets.connect(
            self.settings.openai_ws_url,
            additional_headers=headers,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ) as upstream:
            await self._run_duplex(upstream.send, upstream)

    async def _run_mock(self) -> None:
        upstream = MockRealtimeSession(self.settings.output_sample_rate)
        await upstream.start()
        try:
            await self._run_duplex(upstream.send, upstream.events())
        finally:
            await upstream.close()

    async def _run_omlx(self) -> None:
        upstream = OMLXRealtimeSession(self.settings)
        await upstream.start()
        try:
            await self._run_duplex(upstream.send, upstream.events())
        finally:
            await upstream.close()

    async def _run_duplex(self, upstream_send: Any, upstream_events: Any) -> None:
        client_to_upstream = asyncio.create_task(self._client_to_upstream(upstream_send))
        upstream_to_client = asyncio.create_task(self._upstream_to_client(upstream_events))
        done, pending = await asyncio.wait(
            {client_to_upstream, upstream_to_client},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for task in done:
            exception = task.exception()
            if exception and not isinstance(exception, WebSocketDisconnect):
                raise exception

    async def _client_to_upstream(self, upstream_send: Any) -> None:
        while True:
            raw = await self.client.receive_text()
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await self._send_client_json(
                    {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": "Event must be valid JSON"},
                    }
                )
                continue

            event_type = event.get("type")
            if event_type in {"response.cancel", "output_audio_buffer.clear"}:
                await self.avatar.cancel(self.current_response_id)
                self.timeline.reset()
                await self._send_avatar_state("listening")
            elif event_type == "input_audio_buffer.append":
                await self._send_avatar_state("listening")
            await upstream_send(raw)

    async def _upstream_to_client(self, upstream_events: Any) -> None:
        async for raw in upstream_events:
            event = json.loads(raw)
            event_type = event.get("type")

            if event_type == "input_audio_buffer.barge_in.accepted":
                await self.avatar.cancel(self.current_response_id)
                self.timeline.reset()
                await self._send_avatar_state("listening")
                self.current_response_id = None
            elif event_type == "response.created":
                response = event.get("response") or {}
                self.current_response_id = response.get("id")
                if self.current_response_id:
                    self.timeline.reset(self.current_response_id)
                await self._send_avatar_state("thinking", self.current_response_id)
            elif event_type == "response.output_audio.delta":
                await self._handle_audio_delta(event)
            elif event_type == "response.done":
                await self.avatar.finish(self.current_response_id)
                await self._send_avatar_state("listening", self.current_response_id)
                self.current_response_id = None
            elif event_type == "input_audio_buffer.speech_started":
                await self._send_avatar_state("listening")

            await self._send_client_text(raw)

    async def _avatar_to_client(self) -> None:
        async for event in self.avatar.events():
            event_type = event.get("type", "")
            if not isinstance(event_type, str) or not event_type.startswith("avatar."):
                continue
            await self._send_client_json(event)

    async def _handle_audio_delta(self, event: dict[str, Any]) -> None:
        response_id = event.get("response_id") or self.current_response_id or f"resp_{uuid4().hex[:12]}"
        self.current_response_id = response_id
        try:
            pcm = base64.b64decode(event["delta"], validate=True)
            chunk = self.timeline.ingest(response_id, pcm)
        except (KeyError, ValueError) as error:
            logger.warning("Ignoring invalid audio delta: %s", error)
            return

        await self.avatar.push_audio(chunk)
        await self._send_avatar_state("speaking", response_id)
        await self._send_client_json(
            {
                "type": "avatar.audio.timeline",
                "response_id": response_id,
                "chunk_index": chunk.chunk_index,
                "pts_ms": chunk.pts_ms,
                "duration_ms": chunk.duration_ms,
                "buffered_ms": self.timeline.buffered_duration_ms,
            }
        )

    async def _send_avatar_state(self, state: str, response_id: str | None = None) -> None:
        await self.avatar.set_state(state, response_id=response_id)
        await self._send_client_json(
            {
                "type": "avatar.state",
                "state": state,
                "response_id": response_id,
            }
        )

    async def _send_client_json(self, event: dict[str, Any]) -> None:
        async with self._client_send_lock:
            await self.client.send_json(event)

    async def _send_client_text(self, text: str) -> None:
        async with self._client_send_lock:
            await self.client.send_text(text)
