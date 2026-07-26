from __future__ import annotations

import httpx
import pytest

from virtual_human.xiangongyun import (
    FlashHeadDeployment,
    XiangongyunClient,
    XiangongyunError,
    sanitize_instance,
)


def test_client_authenticates_and_redacts_instance_credentials() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "test-token"
        return httpx.Response(
            200,
            json={
                "code": 200,
                "success": True,
                "data": {
                    "list": [
                        {
                            "id": "gpu-1",
                            "status": "running",
                            "gpu_model": "NVIDIA GeForce RTX 4090 D",
                            "price_per_hour": 1.59,
                            "ssh_domain": "example.invalid",
                            "ssh_port": "2222",
                            "ssh_user": "root",
                            "password": "must-not-leak",
                            "ssh_key": "must-not-leak",
                            "jupyter_token": "must-not-leak",
                            "xgcos_token": "must-not-leak",
                        }
                    ]
                },
            },
        )

    with XiangongyunClient("test-token", transport=httpx.MockTransport(handler)) as client:
        instance = client.instances()[0]

    assert instance["id"] == "gpu-1"
    assert instance["price_per_hour"] == 1.59
    assert "password" not in instance
    assert "ssh_key" not in instance
    assert "jupyter_token" not in instance
    assert "xgcos_token" not in instance


def test_flashhead_deployment_uses_documented_low_cost_defaults() -> None:
    payload = FlashHeadDeployment().as_payload()

    assert payload["gpu_model"] == "NVIDIA GeForce RTX 4090 D"
    assert payload["gpu_count"] == 1
    assert payload["data_center_id"] == 1
    assert payload["image_type"] == "public"
    assert payload["storage"] is False
    assert payload["system_disk_expand"] is False


def test_api_error_does_not_include_token() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"code": 1000, "msg": "invalid access token", "success": False},
        )

    with XiangongyunClient("super-secret", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(XiangongyunError, match="API error 1000") as captured:
            client.balance()

    assert "super-secret" not in str(captured.value)


def test_sanitize_instance_ignores_unknown_sensitive_fields() -> None:
    sanitized = sanitize_instance(
        {
            "id": "gpu-2",
            "name": "renderer",
            "password": "secret",
            "future_private_token": "secret",
        }
    )

    assert sanitized == {"id": "gpu-2", "name": "renderer"}
