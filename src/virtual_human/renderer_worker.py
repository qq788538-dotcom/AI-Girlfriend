from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import shutil
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from virtual_human.mp4_stream import MP4FragmentAssembler

logger = logging.getLogger(__name__)
Emit = Callable[[dict[str, Any]], Awaitable[None]]


class RendererSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    host: str = Field("127.0.0.1", alias="VH_RENDERER_HOST")
    port: int = Field(8770, alias="VH_RENDERER_PORT")
    backend: Literal["mock", "mock-segmented", "liveact-official"] = Field("mock", alias="VH_RENDERER_BACKEND")
    runtime_dir: Path = Field(Path("runtime/renderer"), alias="VH_RENDERER_RUNTIME_DIR")
    public_base_url: str = Field("", alias="VH_RENDERER_PUBLIC_BASE_URL")
    ffmpeg_bin: str = Field("ffmpeg", alias="VH_RENDERER_FFMPEG")
    fps: int = Field(20, alias="VH_RENDERER_FPS")
    cors_origins: str = Field("*", alias="VH_RENDERER_CORS_ORIGINS")
    liveact_demo_url: str = Field("http://127.0.0.1:5001", alias="VH_LIVEACT_DEMO_URL")
    liveact_prompt: str = Field(
        "A beautiful woman is speaking naturally, subtle expression, eye contact, realistic movement.",
        alias="VH_LIVEACT_PROMPT",
    )


@dataclass(slots=True)
class ResponseAudio:
    response_id: str
    sample_rate: int
    pcm16: bytearray = field(default_factory=bytearray)
    next_chunk_index: int = 0

    @property
    def duration_ms(self) -> float:
        return len(self.pcm16) / 2 * 1000 / self.sample_rate

    def append(self, message: dict[str, Any]) -> None:
        chunk_index = int(message["chunk_index"])
        if chunk_index != self.next_chunk_index:
            raise ValueError(f"Expected chunk {self.next_chunk_index}, received {chunk_index}")
        if int(message["sample_rate"]) != self.sample_rate:
            raise ValueError("Sample rate changed inside one response")
        raw = base64.b64decode(message["audio"], validate=True)
        if len(raw) % 2:
            raise ValueError("PCM16 chunk contains a partial sample")
        self.pcm16.extend(raw)
        self.next_chunk_index += 1

    def write_wav(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(self.pcm16)


class RendererBackend(ABC):
    @abstractmethod
    async def render(
        self,
        *,
        session_id: str,
        response: ResponseAudio,
        reference_path: Path,
        output_dir: Path,
        base_url: str,
        emit: Emit,
    ) -> None: ...


class MockVideoBackend(RendererBackend):
    """Produces a real A/V artifact so transport and lifecycle can be tested without CUDA."""

    def __init__(self, ffmpeg_bin: str, fps: int) -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.fps = fps

    async def render(
        self,
        *,
        session_id: str,
        response: ResponseAudio,
        reference_path: Path,
        output_dir: Path,
        base_url: str,
        emit: Emit,
    ) -> None:
        video_path = await self._encode(response, reference_path, output_dir, fragmented=False)
        relative = video_path.relative_to(output_dir.parent.parent)
        await emit(
            {
                "type": "avatar.video.ready",
                "session_id": session_id,
                "response_id": response.response_id,
                "url": f"{base_url}/runtime/{relative.as_posix()}",
                "duration_ms": response.duration_ms,
                "backend": "mock",
                "clock": "audio-master",
                "audio_included": True,
            }
        )
        await emit(
            {
                "type": "avatar.render.done",
                "session_id": session_id,
                "response_id": response.response_id,
                "backend": "mock",
            }
        )

    async def _encode(
        self,
        response: ResponseAudio,
        reference_path: Path,
        output_dir: Path,
        *,
        fragmented: bool,
    ) -> Path:
        wav_path = output_dir / f"{response.response_id}.wav"
        video_path = output_dir / f"{response.response_id}.mp4"
        response.write_wav(wav_path)
        command = [
            self.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-framerate",
            str(self.fps),
            "-i",
            str(reference_path),
            "-i",
            str(wav_path),
            "-vf",
            "scale=416:720:force_original_aspect_ratio=increase,crop=416:720,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "stillimage",
            "-c:a",
            "aac",
            "-shortest",
        ]
        if fragmented:
            command += [
                "-profile:v",
                "baseline",
                "-level:v",
                "3.1",
                "-g",
                str(self.fps),
                "-keyint_min",
                str(self.fps),
                "-sc_threshold",
                "0",
                "-movflags",
                "+frag_keyframe+empty_moov+default_base_moof",
            ]
        else:
            command += ["-movflags", "+faststart"]
        command.append(str(video_path))
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')}")
        return video_path


class MockSegmentedVideoBackend(MockVideoBackend):
    """Exercises the same one-second segment protocol used by the native FlashHead worker."""

    async def render(
        self,
        *,
        session_id: str,
        response: ResponseAudio,
        reference_path: Path,
        output_dir: Path,
        base_url: str,
        emit: Emit,
    ) -> None:
        stream_path = await self._encode(response, reference_path, output_dir, fragmented=True)
        await emit(
            {
                "type": "avatar.media.start",
                "session_id": session_id,
                "response_id": response.response_id,
                "backend": "mock-segmented",
                "mime_type": 'video/mp4; codecs="avc1.42E01F, mp4a.40.2"',
                "fallback_segments": True,
            }
        )
        assembler = MP4FragmentAssembler()
        media_sequence = 0
        for kind, payload in assembler.feed(stream_path.read_bytes()):
            await emit(
                {
                    "type": "avatar.media.init" if kind == "init" else "avatar.media.fragment",
                    "session_id": session_id,
                    "response_id": response.response_id,
                    "sequence": media_sequence,
                    "data": base64.b64encode(payload).decode("ascii"),
                    "backend": "mock-segmented",
                    "mime_type": 'video/mp4; codecs="avc1.42E01F, mp4a.40.2"',
                    "clock": "audio-master",
                    "audio_included": True,
                }
            )
            if kind == "media":
                media_sequence += 1
        assembler.finish()

        segment_bytes = response.sample_rate * 2
        for sequence, offset in enumerate(range(0, len(response.pcm16), segment_bytes)):
            pcm = response.pcm16[offset : offset + segment_bytes]
            segment = ResponseAudio(
                response_id=f"{response.response_id}-{sequence:05d}",
                sample_rate=response.sample_rate,
                pcm16=bytearray(pcm),
            )
            video_path = await self._encode(segment, reference_path, output_dir, fragmented=True)
            relative = video_path.relative_to(output_dir.parent.parent)
            await emit(
                {
                    "type": "avatar.video.segment",
                    "session_id": session_id,
                    "response_id": response.response_id,
                    "sequence": sequence,
                    "url": f"{base_url}/runtime/{relative.as_posix()}",
                    "duration_ms": segment.duration_ms,
                    "backend": "mock-segmented",
                    "clock": "audio-master",
                    "audio_included": True,
                }
            )
        await emit(
            {
                "type": "avatar.render.done",
                "session_id": session_id,
                "response_id": response.response_id,
                "backend": "mock-segmented",
                "segments": (len(response.pcm16) + segment_bytes - 1) // segment_bytes,
            }
        )


class LiveActOfficialBackend(RendererBackend):
    """Adapter for the official SoulX-LiveAct demo server.

    The upstream demo accepts a complete audio file. This adapter is therefore
    a correctness/quality path, not yet the final low-latency incremental path.
    """

    def __init__(self, demo_url: str, fps: int, prompt: str) -> None:
        self.demo_url = demo_url.rstrip("/")
        self.fps = fps
        self.prompt = prompt

    async def render(
        self,
        *,
        session_id: str,
        response: ResponseAudio,
        reference_path: Path,
        output_dir: Path,
        base_url: str,
        emit: Emit,
    ) -> None:
        wav_path = output_dir / f"{response.response_id}.wav"
        response.write_wav(wav_path)
        task_id = f"{session_id}-{response.response_id}"
        async with httpx.AsyncClient(timeout=60) as client:
            with reference_path.open("rb") as image_file, wav_path.open("rb") as audio_file:
                result = await client.post(
                    f"{self.demo_url}/start_stream",
                    data={
                        "task_id": task_id,
                        "main_prompt": self.prompt,
                        "prompt_json": "[]",
                        "fps": str(self.fps),
                        "stream_with_audio": "true",
                    },
                    files={
                        "img_file": (reference_path.name, image_file, "image/jpeg"),
                        "audio_file": (wav_path.name, audio_file, "audio/wav"),
                    },
                )
            result.raise_for_status()

            stream_announced = False
            while True:
                status_response = await client.get(f"{self.demo_url}/task_status/{task_id}")
                status_response.raise_for_status()
                status = status_response.json()
                if status.get("error") or status.get("status") == "failed":
                    raise RuntimeError(status.get("error") or status.get("message") or "LiveAct failed")
                if status.get("stream_ready") and not stream_announced:
                    await emit(
                        {
                            "type": "avatar.stream.ready",
                            "session_id": session_id,
                            "response_id": response.response_id,
                            "url": f"{base_url}/liveact/stream/{task_id}/live.m3u8",
                            "backend": "liveact-official",
                            "clock": "audio-master",
                            "audio_included": True,
                        }
                    )
                    stream_announced = True
                if status.get("is_done"):
                    final_video = Path(str(status.get("final_video_path") or ""))
                    if not final_video.is_file():
                        raise RuntimeError(
                            f"LiveAct completed without a readable final video: {final_video}"
                        )
                    published_video = output_dir / f"{response.response_id}.mp4"
                    shutil.copy2(final_video, published_video)
                    relative = published_video.relative_to(output_dir.parent.parent)
                    await emit(
                        {
                            "type": "avatar.video.ready",
                            "session_id": session_id,
                            "response_id": response.response_id,
                            "url": f"{base_url}/runtime/{relative.as_posix()}",
                            "duration_ms": response.duration_ms,
                            "backend": "liveact-official",
                            "clock": "audio-master",
                            "audio_included": True,
                        }
                    )
                    await emit(
                        {
                            "type": "avatar.render.done",
                            "session_id": session_id,
                            "response_id": response.response_id,
                            "backend": "liveact-official",
                            "metrics": status,
                        }
                    )
                    return
                await asyncio.sleep(0.25)


class RendererSession:
    def __init__(self, settings: RendererSettings, websocket: WebSocket, base_url: str) -> None:
        self.settings = settings
        self.websocket = websocket
        self.base_url = base_url.rstrip("/")
        self.session_id: str | None = None
        self.sample_rate = 24000
        self.session_dir: Path | None = None
        self.reference_path: Path | None = None
        self.responses: dict[str, ResponseAudio] = {}
        self.render_tasks: dict[str, asyncio.Task[None]] = {}
        self._send_lock = asyncio.Lock()
        self.backend = self._build_backend()

    def _build_backend(self) -> RendererBackend:
        if self.settings.backend == "liveact-official":
            return LiveActOfficialBackend(
                self.settings.liveact_demo_url,
                self.settings.fps,
                self.settings.liveact_prompt,
            )
        if self.settings.backend == "mock-segmented":
            return MockSegmentedVideoBackend(self.settings.ffmpeg_bin, self.settings.fps)
        return MockVideoBackend(self.settings.ffmpeg_bin, self.settings.fps)

    async def emit(self, event: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.websocket.send_json(event)

    async def handle(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "avatar.session.start":
            await self._start(message)
        elif message_type == "avatar.audio.append":
            await self._append(message)
        elif message_type == "avatar.response.finish":
            await self._finish(message.get("response_id"))
        elif message_type == "avatar.response.cancel":
            await self._cancel(message.get("response_id"))
        elif message_type == "avatar.state":
            await self.emit(
                {
                    "type": "avatar.state.ack",
                    "state": message.get("state"),
                    "response_id": message.get("response_id"),
                }
            )
        else:
            raise ValueError(f"Unsupported renderer message: {message_type}")

    async def _start(self, message: dict[str, Any]) -> None:
        self.session_id = str(message["session_id"])
        self.sample_rate = int(message["sample_rate"])
        self.session_dir = self.settings.runtime_dir / "sessions" / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        encoded_reference = message.get("reference_image")
        if encoded_reference:
            reference_name = Path(message.get("reference_name") or "reference.jpg").name
            self.reference_path = self.session_dir / reference_name
            self.reference_path.write_bytes(base64.b64decode(encoded_reference, validate=True))
        else:
            source = Path(str(message.get("reference", "")))
            if not source.is_file():
                raise ValueError("Renderer requires reference_image or a readable reference path")
            self.reference_path = self.session_dir / source.name
            shutil.copy2(source, self.reference_path)

        await self.emit(
            {
                "type": "avatar.session.ready",
                "session_id": self.session_id,
                "backend": self.settings.backend,
                "fps": self.settings.fps,
            }
        )

    async def _append(self, message: dict[str, Any]) -> None:
        if self.session_id is None:
            raise ValueError("avatar.session.start is required before audio")
        response_id = str(message["response_id"])
        response = self.responses.setdefault(response_id, ResponseAudio(response_id, self.sample_rate))
        response.append(message)
        await self.emit(
            {
                "type": "avatar.audio.ack",
                "response_id": response_id,
                "chunk_index": int(message["chunk_index"]),
                "buffered_ms": response.duration_ms,
            }
        )

    async def _finish(self, response_id: str | None) -> None:
        if not response_id or response_id not in self.responses:
            return
        if response_id in self.render_tasks and not self.render_tasks[response_id].done():
            return
        response = self.responses[response_id]
        await self.emit(
            {
                "type": "avatar.render.accepted",
                "response_id": response_id,
                "duration_ms": response.duration_ms,
                "backend": self.settings.backend,
            }
        )
        task = asyncio.create_task(self._render(response))
        self.render_tasks[response_id] = task

    async def _render(self, response: ResponseAudio) -> None:
        assert self.session_id is not None
        assert self.session_dir is not None
        assert self.reference_path is not None
        try:
            await self.backend.render(
                session_id=self.session_id,
                response=response,
                reference_path=self.reference_path,
                output_dir=self.session_dir,
                base_url=self.base_url,
                emit=self.emit,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception("Avatar rendering failed")
            await self.emit(
                {
                    "type": "avatar.error",
                    "response_id": response.response_id,
                    "message": str(error),
                    "backend": self.settings.backend,
                }
            )

    async def _cancel(self, response_id: str | None) -> None:
        if response_id:
            task = self.render_tasks.pop(response_id, None)
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            self.responses.pop(response_id, None)
        await self.emit({"type": "avatar.response.cancelled", "response_id": response_id})

    async def close(self) -> None:
        for task in self.render_tasks.values():
            if not task.done():
                task.cancel()
        for task in self.render_tasks.values():
            with contextlib.suppress(asyncio.CancelledError):
                await task


def create_renderer_app(settings: RendererSettings | None = None) -> FastAPI:
    renderer_settings = settings or RendererSettings()
    renderer_settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    application = FastAPI(title="Virtual Human GPU Renderer", version="0.1.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in renderer_settings.cors_origins.split(",") if origin.strip()],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @application.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "backend": renderer_settings.backend,
                "fps": renderer_settings.fps,
                "runtime_dir": str(renderer_settings.runtime_dir),
                "liveact_demo_url": (
                    renderer_settings.liveact_demo_url if renderer_settings.backend == "liveact-official" else None
                ),
            }
        )

    @application.get("/liveact/stream/{task_id}/{filename:path}")
    async def liveact_stream(task_id: str, filename: str) -> Response:
        if renderer_settings.backend != "liveact-official":
            return Response(status_code=404)
        async with httpx.AsyncClient(timeout=30) as client:
            upstream = await client.get(
                f"{renderer_settings.liveact_demo_url}/stream/{task_id}/{filename}"
            )
        suffix = Path(filename).suffix.lower()
        media_type = {
            ".m3u8": "application/vnd.apple.mpegurl",
            ".ts": "video/mp2t",
        }.get(suffix, upstream.headers.get("content-type"))
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=media_type,
            headers={"Cache-Control": upstream.headers.get("cache-control", "no-cache")},
        )

    @application.websocket("/avatar")
    async def avatar(websocket: WebSocket) -> None:
        await websocket.accept()
        scheme = "https" if websocket.url.scheme == "wss" else "http"
        base_url = renderer_settings.public_base_url or f"{scheme}://{websocket.url.netloc}"
        session = RendererSession(renderer_settings, websocket, base_url)
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    message = json.loads(raw)
                    if not isinstance(message, dict):
                        raise ValueError("Renderer message must be a JSON object")
                    await session.handle(message)
                except (KeyError, ValueError, TypeError) as error:
                    await session.emit({"type": "avatar.error", "message": str(error)})
        except WebSocketDisconnect:
            pass
        finally:
            await session.close()

    application.mount(
        "/runtime",
        StaticFiles(directory=renderer_settings.runtime_dir),
        name="runtime",
    )
    return application


app = create_renderer_app()


def main() -> None:
    settings = RendererSettings()
    uvicorn.run(
        "virtual_human.renderer_worker:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
