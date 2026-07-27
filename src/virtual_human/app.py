from __future__ import annotations

import logging
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from virtual_human.avatar import build_avatar_sink
from virtual_human.bridge import RealtimeBridge
from virtual_human.config import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = Settings()
app = FastAPI(title="Virtual Human Realtime Gateway", version="0.1.0")


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "offline_runtime": settings.offline_runtime,
            "upstream_mode": settings.upstream_mode,
            "chat_backend": settings.chat_backend,
            "model": (
                settings.openai_realtime_model
                if settings.upstream_mode == "openai"
                else settings.resolved_chat_model
                if settings.upstream_mode == "omlx"
                else "virtual-human-mock"
            ),
            "avatar_backend": settings.avatar_backend,
            "avatar_renderer": "remote" if settings.avatar_renderer_ws else "local-fallback",
            "playback_owner": "renderer" if settings.avatar_renderer_ws else "browser",
            "audio_clock": "audio-master",
            "sample_rate": settings.output_sample_rate,
            "preroll_ms": settings.av_sync_preroll_ms,
            "avatar_identity": {
                "locked": settings.avatar_reference_locked,
                "reference": Path(settings.avatar_reference).name,
            },
            "voice_profile": {
                "locked": settings.voice_profile_locked,
                "model": settings.omlx_tts_model,
                "backend": settings.resolved_tts_base_url,
                "protocol": settings.tts_protocol,
                "flow": settings.tts_flow_mode,
                "seed": settings.tts_seed,
            },
            "asr": {
                "model": settings.omlx_stt_model,
                "backend": settings.resolved_asr_base_url,
            },
            "memory": {
                "enabled": settings.memory_enabled,
                "backend": "openviking" if settings.memory_enabled else "disabled",
                "base_url": settings.memory_base_url if settings.memory_enabled else "",
                "agent_id": settings.memory_agent_id if settings.memory_enabled else "",
            },
        }
    )


@app.get("/client-config")
async def client_config() -> JSONResponse:
    return JSONResponse(
        {
            "model": (
                settings.openai_realtime_model
                if settings.upstream_mode == "openai"
                else settings.resolved_chat_model
            ),
            "voice": (
                settings.openai_voice
                if settings.upstream_mode == "openai"
                else settings.omlx_tts_voice
            ),
            "vad_eagerness": settings.vad_eagerness,
            "barge_in_enabled": settings.barge_in_enabled,
            "chat_backend": settings.chat_backend,
            "persona_name": settings.persona_name,
            "instructions": settings.persona_instructions,
        }
    )


@app.websocket("/v1/realtime")
async def realtime(websocket: WebSocket) -> None:
    await websocket.accept()
    cloud_avatar_enabled = websocket.query_params.get("avatar", "1").lower() not in {
        "0",
        "false",
        "off",
    }
    avatar = build_avatar_sink(
        settings.avatar_renderer_ws if cloud_avatar_enabled else "",
        settings.avatar_backend,
        settings.av_sync_preroll_ms,
        settings.resolved_avatar_renderer_token() if cloud_avatar_enabled else "",
    )
    bridge = RealtimeBridge(settings, websocket, avatar)
    try:
        await bridge.run()
    except Exception:
        logger.exception("Realtime session failed")
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "error": {"type": "gateway_error", "message": "Realtime gateway session failed"},
                }
            )
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/avatar-media/{media_path:path}")
async def avatar_media(media_path: str, request: Request) -> Response:
    """Same-origin proxy for loopback-only LiveAct HLS and final MP4 files."""
    if not settings.avatar_media_base_url:
        raise HTTPException(status_code=404)
    if not media_path or ".." in Path(media_path).parts:
        raise HTTPException(status_code=400, detail="Invalid avatar media path")

    headers: dict[str, str] = {}
    if byte_range := request.headers.get("range"):
        headers["Range"] = byte_range
    async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
        upstream = await client.get(
            f"{settings.avatar_media_base_url.rstrip('/')}/{media_path}",
            headers=headers,
        )
    response_headers = {
        name: value
        for name, value in upstream.headers.items()
        if name.lower() in {"accept-ranges", "cache-control", "content-length", "content-range"}
    }
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
        headers=response_headers,
    )


public_dir = Path(settings.public_dir).resolve()
if not public_dir.exists():
    raise RuntimeError(f"Public directory does not exist: {public_dir}")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(public_dir / "index.html")


app.mount("/", StaticFiles(directory=public_dir, html=True), name="public")


def main() -> None:
    uvicorn.run("virtual_human.app:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
