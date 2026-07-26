from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

FragmentKind = Literal["init", "media"]
EmitFragment = Callable[[FragmentKind, bytes], Awaitable[None]]
GENTLE_UNSHARP_FILTER = "unsharp=5:5:0.55:3:3:0.15"


def build_video_filter(
    *,
    input_width: int,
    input_height: int,
    output_width: int,
    output_height: int,
    sharpen: bool,
) -> str | None:
    filters: list[str] = []
    if (output_width, output_height) != (input_width, input_height):
        filters.append(f"scale={output_width}:{output_height}:flags=lanczos")
    if sharpen:
        filters.append(GENTLE_UNSHARP_FILTER)
    return ",".join(filters) or None


@dataclass(slots=True)
class MP4BoxParser:
    buffer: bytearray = field(default_factory=bytearray)

    def feed(self, data: bytes) -> list[tuple[bytes, bytes]]:
        self.buffer.extend(data)
        boxes: list[tuple[bytes, bytes]] = []
        while len(self.buffer) >= 8:
            size = int.from_bytes(self.buffer[:4], "big")
            box_type = bytes(self.buffer[4:8])
            header_size = 8
            if size == 1:
                if len(self.buffer) < 16:
                    break
                size = int.from_bytes(self.buffer[8:16], "big")
                header_size = 16
            elif size == 0:
                break
            if size < header_size:
                raise ValueError(f"Invalid MP4 box size {size} for {box_type!r}")
            if len(self.buffer) < size:
                break
            box = bytes(self.buffer[:size])
            del self.buffer[:size]
            boxes.append((box_type, box))
        return boxes

    def finish(self) -> None:
        if self.buffer:
            raise ValueError(f"Incomplete MP4 stream: {len(self.buffer)} trailing bytes")


@dataclass(slots=True)
class MP4FragmentAssembler:
    parser: MP4BoxParser = field(default_factory=MP4BoxParser)
    init_complete: bool = False
    init_parts: list[bytes] = field(default_factory=list)
    media_parts: list[bytes] = field(default_factory=list)

    def feed(self, data: bytes) -> list[tuple[FragmentKind, bytes]]:
        outputs: list[tuple[FragmentKind, bytes]] = []
        for box_type, box in self.parser.feed(data):
            if not self.init_complete:
                self.init_parts.append(box)
                if box_type == b"moov":
                    outputs.append(("init", b"".join(self.init_parts)))
                    self.init_parts.clear()
                    self.init_complete = True
                continue
            if box_type in {b"mfra", b"mfro"}:
                continue
            self.media_parts.append(box)
            if box_type == b"mdat":
                outputs.append(("media", b"".join(self.media_parts)))
                self.media_parts.clear()
        return outputs

    def finish(self) -> None:
        self.parser.finish()


class ContinuousMP4Muxer:
    """Muxes raw RGB frames and PCM into one fragmented MP4 encoder session."""

    def __init__(
        self,
        *,
        ffmpeg_bin: str,
        width: int,
        height: int,
        fps: int,
        sample_rate: int,
        gop_size: int,
        emit: EmitFragment,
        crf: int = 18,
        sharpen: bool = True,
        output_width: int | None = None,
        output_height: int | None = None,
    ) -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.width = width
        self.height = height
        self.fps = fps
        self.sample_rate = sample_rate
        self.gop_size = gop_size
        self.emit = emit
        self.crf = crf
        self.sharpen = sharpen
        self.output_width = output_width or width
        self.output_height = output_height or height
        self.process: asyncio.subprocess.Process | None = None
        self.audio_write_fd: int | None = None
        self.reader_task: asyncio.Task[None] | None = None
        self.stderr_task: asyncio.Task[bytes] | None = None

    async def start(self) -> None:
        if self.process is not None:
            return
        audio_read_fd, self.audio_write_fd = os.pipe()
        video_filter = build_video_filter(
            input_width=self.width,
            input_height=self.height,
            output_width=self.output_width,
            output_height=self.output_height,
            sharpen=self.sharpen,
        )
        command = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-f",
            "s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-i",
            f"pipe:{audio_read_fd}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            *(["-vf", video_filter] if video_filter else []),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            str(self.crf),
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-level:v",
            "3.1",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(self.gop_size),
            "-keyint_min",
            str(self.gop_size),
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            "-movflags",
            "+frag_keyframe+empty_moov+default_base_moof+dash",
            "-frag_duration",
            str(round(self.gop_size * 1_000_000 / self.fps)),
            "-f",
            "mp4",
            "pipe:1",
        ]
        try:
            self.process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                pass_fds=(audio_read_fd,),
            )
        finally:
            os.close(audio_read_fd)
        self.reader_task = asyncio.create_task(self._read_fragments())
        self.stderr_task = asyncio.create_task(self._read_stderr())

    async def write(self, rgb24: bytes, pcm16: bytes) -> None:
        await self.start()
        assert self.process is not None
        assert self.process.stdin is not None
        assert self.audio_write_fd is not None
        self.process.stdin.write(rgb24)
        await asyncio.gather(
            self.process.stdin.drain(),
            self._write_audio(pcm16),
        )

    async def finish(self) -> None:
        if self.process is None:
            return
        assert self.process.stdin is not None
        self.process.stdin.close()
        # Both inputs must receive EOF before FFmpeg can honor `-shortest` and
        # exit. Waiting for the video pipe first can deadlock while FFmpeg is
        # still waiting on the audio pipe.
        if self.audio_write_fd is not None:
            os.close(self.audio_write_fd)
            self.audio_write_fd = None
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            await self.process.stdin.wait_closed()
        return_code = await self.process.wait()
        if self.reader_task:
            await self.reader_task
        stderr = await self.stderr_task if self.stderr_task else b""
        self.process = None
        if return_code:
            raise RuntimeError(f"continuous ffmpeg failed: {stderr.decode(errors='replace')}")

    async def cancel(self) -> None:
        if self.process is None:
            return
        if self.audio_write_fd is not None:
            os.close(self.audio_write_fd)
            self.audio_write_fd = None
        self.process.terminate()
        with contextlib.suppress(ProcessLookupError):
            await self.process.wait()
        for task in (self.reader_task, self.stderr_task):
            if task and not task.done():
                task.cancel()
        self.process = None

    async def _write_audio(self, data: bytes) -> None:
        if self.audio_write_fd is None:
            raise RuntimeError("Audio pipe is closed")
        view = memoryview(data)
        while view:
            written = await asyncio.to_thread(os.write, self.audio_write_fd, view)
            view = view[written:]

    async def _read_fragments(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        assembler = MP4FragmentAssembler()
        while chunk := await self.process.stdout.read(64 * 1024):
            for kind, payload in assembler.feed(chunk):
                await self.emit(kind, payload)
        assembler.finish()

    async def _read_stderr(self) -> bytes:
        assert self.process is not None
        assert self.process.stderr is not None
        return await self.process.stderr.read()
