from __future__ import annotations

from fastapi.testclient import TestClient

from virtual_human.embedding_server import create_embedding_app


class FakeEmbeddingBackend:
    model_name = "local-qwen-embedding"
    dimension = 3

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 0.5, 1.0] for index, _text in enumerate(texts)]


def test_embedding_server_exposes_openai_compatible_shape() -> None:
    client = TestClient(create_embedding_app(FakeEmbeddingBackend()))

    response = client.post(
        "/v1/embeddings",
        json={
            "model": "Qwen3-Embedding-0.6B",
            "input": ["用户喜欢黑咖啡。", "用户叫小猫。"],
            "encoding_format": "float",
            "dimensions": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "Qwen3-Embedding-0.6B"
    assert payload["data"] == [
        {"object": "embedding", "index": 0, "embedding": [0.0, 0.5, 1.0]},
        {"object": "embedding", "index": 1, "embedding": [1.0, 0.5, 1.0]},
    ]
    assert client.get("/healthz").json() == {
        "status": "ok",
        "model": "local-qwen-embedding",
        "dimension": 3,
        "device": "cpu",
        "local_files_only": True,
    }


def test_embedding_server_rejects_shape_changes_and_base64() -> None:
    client = TestClient(create_embedding_app(FakeEmbeddingBackend()))

    assert (
        client.post(
            "/v1/embeddings",
            json={"input": "test", "dimensions": 2},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/v1/embeddings",
            json={"input": "test", "encoding_format": "base64"},
        ).status_code
        == 400
    )
