from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class EmbeddingBackend(Protocol):
    model_name: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerBackend:
    """CPU-only, local-files-only embedding backend for Xiangongyun."""

    def __init__(self, model_path: str, *, max_sequence_length: int = 2048) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_path
        self._model = SentenceTransformer(
            model_path,
            device="cpu",
            trust_remote_code=True,
            local_files_only=True,
        )
        self._model.max_seq_length = max_sequence_length
        dimension = self._model.get_sentence_embedding_dimension()
        if not dimension:
            raise RuntimeError("Embedding model did not report an output dimension")
        self.dimension = int(dimension)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=min(16, max(1, len(texts))),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.astype("float32", copy=False).tolist() for vector in vectors]


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    encoding_format: str = "float"
    dimensions: int | None = Field(default=None, ge=1)


def create_embedding_app(backend: EmbeddingBackend) -> FastAPI:
    app = FastAPI(title="Local OpenAI-compatible Embedding Server", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": backend.model_name,
            "dimension": backend.dimension,
            "device": "cpu",
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

    @app.post("/v1/embeddings")
    async def embeddings(request: EmbeddingRequest) -> dict[str, Any]:
        if request.encoding_format != "float":
            raise HTTPException(status_code=400, detail="Only encoding_format=float is supported")
        texts = [request.input] if isinstance(request.input, str) else request.input
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise HTTPException(status_code=400, detail="input must contain non-empty strings")
        if len(texts) > 128:
            raise HTTPException(status_code=400, detail="A maximum of 128 inputs is supported")
        if request.dimensions not in {None, backend.dimension}:
            raise HTTPException(
                status_code=400,
                detail=f"This model has a fixed dimension of {backend.dimension}",
            )

        vectors = await asyncio.to_thread(backend.encode, texts)
        if len(vectors) != len(texts) or any(
            len(vector) != backend.dimension for vector in vectors
        ):
            raise HTTPException(status_code=500, detail="Embedding backend returned an invalid shape")
        return {
            "object": "list",
            "model": request.model or backend.model_name,
            "data": [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": vector,
                }
                for index, vector in enumerate(vectors)
            ],
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only local embedding server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    args = parser.parse_args()

    backend = SentenceTransformerBackend(
        args.model,
        max_sequence_length=args.max_sequence_length,
    )
    uvicorn.run(
        create_embedding_app(backend),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
