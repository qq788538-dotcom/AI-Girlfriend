#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import wave
from pathlib import Path

import websockets


def load_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav_file:
        audio_format = (
            wav_file.getframerate(),
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
        )
        if audio_format != (24_000, 1, 2):
            raise ValueError(
                f"Input WAV must be 24kHz mono PCM16; received {audio_format}"
            )
        return wav_file.readframes(wav_file.getnframes())


async def benchmark(
    url: str,
    timeout: float,
    audio_path: Path | None,
    *,
    require_render: bool,
    output_media: Path | None,
    output_audio: Path | None,
    memory_enabled: bool,
    inject_output_echo_ms: int,
    playback_preroll_ms: int,
) -> dict[str, object]:
    started = time.perf_counter()
    timings: dict[str, float] = {}
    counts = {"audio_chunks": 0, "media_fragments": 0}
    event_counts: dict[str, int] = {}
    transcript = ""
    input_transcript = ""
    response_done = False
    render_done = False
    media_bytes = bytearray()
    audio_bytes = bytearray()
    last_audio_arrival_ms: float | None = None
    max_audio_arrival_gap_ms = 0.0
    simulated_playback_end_ms: float | None = None
    playback_underruns: list[float] = []
    echo_audio = bytearray()
    echo_injected = False
    echo_decision = ""
    echo_target_bytes = 24_000 * 2 * inject_output_echo_ms // 1000

    async with websockets.connect(url, max_size=None, ping_timeout=180) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "memory": {"enabled": memory_enabled},
                        "metadata": {"purpose": "benchmark"},
                    },
                }
            )
        )
        if audio_path is None:
            await websocket.send(json.dumps({"type": "response.create"}))
        else:
            pcm16 = load_pcm16(audio_path)
            chunk_bytes = 24_000 * 2 * 40 // 1000
            for offset in range(0, len(pcm16), chunk_bytes):
                await websocket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(
                                pcm16[offset : offset + chunk_bytes]
                            ).decode("ascii"),
                        }
                    )
                )
            await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))

        async def receive_until_complete() -> None:
            nonlocal transcript, input_transcript, response_done, render_done
            nonlocal echo_injected, echo_decision
            nonlocal last_audio_arrival_ms, max_audio_arrival_gap_ms
            nonlocal simulated_playback_end_ms
            async for raw in websocket:
                if isinstance(raw, bytes):
                    continue
                event = json.loads(raw)
                event_type = str(event.get("type") or "unknown")
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                elapsed_ms = (time.perf_counter() - started) * 1000

                if event_type == "response.created":
                    timings.setdefault("response_created_ms", elapsed_ms)
                elif event_type == "response.output_audio_transcript.done":
                    timings.setdefault("transcript_ready_ms", elapsed_ms)
                    transcript = str(event.get("transcript") or "")
                elif (
                    event_type
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    timings.setdefault("input_transcript_ready_ms", elapsed_ms)
                    input_transcript = str(event.get("transcript") or "")
                elif event_type == "response.output_audio.delta":
                    timings.setdefault("first_audio_delta_ms", elapsed_ms)
                    counts["audio_chunks"] += 1
                    delta = event.get("delta")
                    pcm = (
                        base64.b64decode(str(delta), validate=True)
                        if delta
                        else b""
                    )
                    if last_audio_arrival_ms is not None:
                        max_audio_arrival_gap_ms = max(
                            max_audio_arrival_gap_ms,
                            elapsed_ms - last_audio_arrival_ms,
                        )
                    last_audio_arrival_ms = elapsed_ms
                    chunk_duration_ms = len(pcm) / 2 * 1000 / 24_000
                    if simulated_playback_end_ms is None:
                        simulated_playback_end_ms = elapsed_ms + playback_preroll_ms
                    elif elapsed_ms > simulated_playback_end_ms:
                        playback_underruns.append(elapsed_ms - simulated_playback_end_ms)
                        simulated_playback_end_ms = elapsed_ms + 60
                    simulated_playback_end_ms += chunk_duration_ms
                    if output_audio and delta:
                        audio_bytes.extend(pcm)
                    if inject_output_echo_ms > 0 and not echo_injected:
                        if delta:
                            echo_audio.extend(pcm)
                        if len(echo_audio) >= echo_target_bytes:
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "input_audio_buffer.barge_in.append",
                                        "audio": base64.b64encode(
                                            bytes(echo_audio)
                                        ).decode("ascii"),
                                    }
                                )
                            )
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "input_audio_buffer.barge_in.commit",
                                    }
                                )
                            )
                            echo_injected = True
                            timings.setdefault("echo_injected_ms", elapsed_ms)
                elif event_type == "input_audio_buffer.barge_in.echo_ignored":
                    echo_decision = "ignored"
                    timings.setdefault("echo_decision_ms", elapsed_ms)
                elif event_type == "input_audio_buffer.barge_in.accepted":
                    echo_decision = "accepted"
                    timings.setdefault("echo_decision_ms", elapsed_ms)
                elif event_type == "avatar.media.fragment":
                    timings.setdefault("first_media_fragment_ms", elapsed_ms)
                    counts["media_fragments"] += 1
                    if output_media and event.get("data"):
                        media_bytes.extend(
                            base64.b64decode(event["data"], validate=True)
                        )
                elif event_type == "avatar.media.init":
                    if output_media and event.get("data"):
                        media_bytes.extend(
                            base64.b64decode(event["data"], validate=True)
                        )
                elif event_type == "response.done":
                    timings.setdefault("response_done_ms", elapsed_ms)
                    response_done = True
                elif event_type == "avatar.render.done":
                    timings.setdefault("render_done_ms", elapsed_ms)
                    render_done = True
                elif event_type in {"error", "avatar.error"}:
                    raise RuntimeError(
                        event.get("error") or event.get("message") or event_type
                    )

                echo_complete = inject_output_echo_ms <= 0 or bool(echo_decision)
                if (
                    response_done
                    and (render_done or not require_render)
                    and echo_complete
                ):
                    break

        try:
            await asyncio.wait_for(receive_until_complete(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

    if output_media and media_bytes:
        output_media.parent.mkdir(parents=True, exist_ok=True)
        output_media.write_bytes(media_bytes)
    if output_audio and audio_bytes:
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_audio), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24_000)
            wav_file.writeframes(audio_bytes)

    return {
        "status": (
            "ok"
            if (
                response_done
                and (render_done or not require_render)
                and (inject_output_echo_ms <= 0 or echo_decision == "ignored")
            )
            else "timeout"
        ),
        "url": url,
        "memory_enabled": memory_enabled,
        "timings": timings,
        "counts": counts,
        "audio_delivery": {
            "simulated_preroll_ms": playback_preroll_ms,
            "max_arrival_gap_ms": round(max_audio_arrival_gap_ms, 3),
            "simulated_playback_underruns": len(playback_underruns),
            "max_simulated_underrun_ms": round(max(playback_underruns, default=0.0), 3),
        },
        "event_counts": event_counts,
        "input_transcript": input_transcript,
        "transcript": transcript,
        "output_media": str(output_media) if output_media and media_bytes else "",
        "output_media_bytes": len(media_bytes),
        "output_audio": str(output_audio) if output_audio and audio_bytes else "",
        "output_audio_bytes": len(audio_bytes),
        "echo_test": {
            "requested_ms": inject_output_echo_ms,
            "injected": echo_injected,
            "decision": echo_decision,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the complete virtual-human gateway.")
    parser.add_argument("--url", default="ws://127.0.0.1:8765/v1/realtime")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--audio", type=Path)
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Do not wait for a remote avatar render completion event.",
    )
    parser.add_argument("--output-media", type=Path)
    parser.add_argument("--output-audio", type=Path)
    parser.add_argument(
        "--playback-preroll-ms",
        type=int,
        default=1000,
        help="Browser PCM pre-roll used by the underrun simulation.",
    )
    parser.add_argument(
        "--with-memory",
        action="store_true",
        help=(
            "Allow benchmark turns to read and write the girlfriend memory. "
            "Disabled by default so automated tests cannot pollute real memories."
        ),
    )
    parser.add_argument(
        "--inject-output-echo-ms",
        type=int,
        default=0,
        help=(
            "During playback, feed this many milliseconds of synthesized output "
            "back through the barge-in path. The run passes only when it is "
            "classified as playback echo instead of cancelling the answer."
        ),
    )
    args = parser.parse_args()
    result = asyncio.run(
        benchmark(
            args.url,
            args.timeout,
            args.audio,
            require_render=not args.audio_only,
            output_media=args.output_media,
            output_audio=args.output_audio,
            memory_enabled=args.with_memory,
            inject_output_echo_ms=args.inject_output_echo_ms,
            playback_preroll_ms=args.playback_preroll_ms,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
