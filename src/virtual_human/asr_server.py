from __future__ import annotations

import argparse
import asyncio
import io
import threading
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile


class ASRBackend(Protocol):
    model_name: str

    def transcribe(self, audio: bytes, language: str | None = None) -> str: ...


class QwenASRBackend:
    """Resident Qwen3-ASR backend for complete, single-turn utterances."""

    def __init__(self, model_path: str) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        self.model_name = model_path
        self._lock = threading.Lock()
        self._model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=1,
            max_new_tokens=256,
        )

    def transcribe(self, audio: bytes, language: str | None = None) -> str:
        import numpy as np
        import soundfile as sf

        samples, sample_rate = sf.read(
            io.BytesIO(audio),
            dtype="float32",
            always_2d=False,
        )
        if getattr(samples, "ndim", 1) > 1:
            samples = np.mean(samples, axis=1, dtype=np.float32)
        with self._lock:
            results = self._model.transcribe(
                audio=(samples, int(sample_rate)),
                language=language or None,
            )
        if not results:
            return ""
        return str(getattr(results[0], "text", "") or "").strip()


def create_asr_app(backend: ASRBackend) -> FastAPI:
    app = FastAPI(title="Local OpenAI-compatible Qwen3 ASR", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": backend.model_name,
            "device": "cuda:0",
            "local_files_only": True,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": backend.model_name,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form("Qwen3-ASR-0.6B"),
        language: str | None = Form(None),
    ) -> dict[str, str]:
        del model
        audio = await file.read()
        if not audio:
            raise HTTPException(status_code=400, detail="audio file is empty")
        if len(audio) > 32 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="audio file is too large")
        try:
            text = await asyncio.to_thread(backend.transcribe, audio, language)
        except Exception as error:
            raise HTTPException(status_code=422, detail=f"transcription failed: {error}") from error
        return {"text": text}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Qwen3-ASR server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    uvicorn.run(
        create_asr_app(QwenASRBackend(args.model)),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
