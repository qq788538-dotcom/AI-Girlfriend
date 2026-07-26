from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_API_BASE = "https://api.xiangongyun.com"
DEFAULT_DATA_CENTER_ID = 1
DEFAULT_FLASHHEAD_GPU = "NVIDIA GeForce RTX 4090 D"
DEFAULT_FLASHHEAD_IMAGE = "02b564d3-9510-4f80-b56d-409900dab8da"

_SAFE_INSTANCE_FIELDS = (
    "id",
    "name",
    "status",
    "progress",
    "data_center_name",
    "gpu_model",
    "gpu_used",
    "cpu_model",
    "cpu_core_count",
    "memory_size",
    "system_disk_size",
    "data_disk_size",
    "price_per_hour",
    "base_price",
    "image_price",
    "retain_price",
    "public_image",
    "image_id",
    "image_type",
    "ssh_domain",
    "ssh_port",
    "ssh_user",
    "web_url",
    "create_timestamp",
    "start_timestamp",
    "stop_timestamp",
)


class XiangongyunError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FlashHeadDeployment:
    gpu_model: str = DEFAULT_FLASHHEAD_GPU
    gpu_count: int = 1
    data_center_id: int = DEFAULT_DATA_CENTER_ID
    image: str = DEFAULT_FLASHHEAD_IMAGE
    image_type: str = "public"
    name: str = "virtual-human-flashhead-lite"

    def as_payload(self) -> dict[str, Any]:
        return {
            "gpu_model": self.gpu_model,
            "gpu_count": self.gpu_count,
            "data_center_id": self.data_center_id,
            "image": self.image,
            "image_type": self.image_type,
            "storage": False,
            "system_disk_expand": False,
            "name": self.name,
        }


def sanitize_instance(instance: Mapping[str, Any]) -> dict[str, Any]:
    """Return instance metadata without passwords, keys, or access tokens."""
    return {field: instance[field] for field in _SAFE_INSTANCE_FIELDS if field in instance}


class XiangongyunClient:
    def __init__(
        self,
        token: str,
        *,
        base_url: str = DEFAULT_API_BASE,
        timeout: float = 30,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not token.strip():
            raise XiangongyunError("XGC_API_TOKEN is empty")
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": token.strip(), "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "XiangongyunClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
    ) -> Any:
        try:
            response = self._client.request(method, path, json=body)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise XiangongyunError(f"Xiangongyun request failed: {error}") from error
        if not isinstance(payload, dict):
            raise XiangongyunError("Xiangongyun returned a non-object response")
        if not payload.get("success"):
            code = payload.get("code", "unknown")
            message = payload.get("msg", "unknown API error")
            raise XiangongyunError(f"Xiangongyun API error {code}: {message}")
        return payload.get("data")

    def balance(self) -> float:
        data = self._request("GET", "/open/balance")
        if not isinstance(data, dict) or "balance" not in data:
            raise XiangongyunError("Balance response is missing data.balance")
        return float(data["balance"])

    def instances(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/open/instances")
        if not isinstance(data, dict) or not isinstance(data.get("list"), list):
            raise XiangongyunError("Instance response is missing data.list")
        return [sanitize_instance(item) for item in data["list"] if isinstance(item, dict)]

    def instance(self, instance_id: str) -> dict[str, Any]:
        data = self._request("GET", f"/open/instance/{instance_id}")
        if not isinstance(data, dict):
            raise XiangongyunError("Instance response is missing data")
        return sanitize_instance(data)

    def deploy(self, deployment: FlashHeadDeployment) -> str:
        data = self._request("POST", "/open/instance/deploy", body=deployment.as_payload())
        if not isinstance(data, dict) or not data.get("id"):
            raise XiangongyunError("Deploy response is missing data.id")
        return str(data["id"])

    def action(self, instance_id: str, action: str) -> Any:
        allowed = {"boot", "shutdown_release_gpu", "shutdown_destroy"}
        if action not in allowed:
            raise XiangongyunError(f"Unsupported instance action: {action}")
        return self._request("POST", f"/open/instance/{action}", body={"id": instance_id})

    def wait_for_status(
        self,
        instance_id: str,
        *,
        status: str,
        timeout_seconds: float,
        poll_seconds: float = 5,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            instance = self.instance(instance_id)
            if instance.get("status") == status:
                return instance
            if time.monotonic() >= deadline:
                current = instance.get("status", "unknown")
                raise XiangongyunError(
                    f"Timed out waiting for {instance_id} to become {status}; current status is {current}"
                )
            time.sleep(poll_seconds)


def _token_from_environment() -> str:
    token = os.environ.get("XGC_API_TOKEN", "")
    if not token:
        raise XiangongyunError(
            "Set XGC_API_TOKEN in the current shell. The CLI intentionally does not accept tokens as arguments."
        )
    return token


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely inspect and manage Xiangongyun GPU instances for the virtual-human renderer."
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("balance", help="Show account balance.")
    commands.add_parser("instances", help="List instances with all credentials redacted.")

    instance = commands.add_parser("instance", help="Show one sanitized instance.")
    instance.add_argument("instance_id")

    deploy = commands.add_parser("deploy-flashhead", help="Create a billed FlashHead Lite GPU instance.")
    deploy.add_argument("--gpu-model", default=DEFAULT_FLASHHEAD_GPU)
    deploy.add_argument("--gpu-count", type=int, default=1)
    deploy.add_argument("--data-center-id", type=int, default=DEFAULT_DATA_CENTER_ID)
    deploy.add_argument("--image", default=DEFAULT_FLASHHEAD_IMAGE)
    deploy.add_argument("--image-type", choices=("public", "community", "private"), default="public")
    deploy.add_argument("--name", default="virtual-human-flashhead-lite")
    deploy.add_argument(
        "--confirm-billing",
        action="store_true",
        help="Required acknowledgement that the API call starts a billable instance.",
    )

    wait = commands.add_parser("wait", help="Wait for an instance status.")
    wait.add_argument("instance_id")
    wait.add_argument("--status", default="running")
    wait.add_argument("--timeout", type=float, default=900)

    for action in ("boot", "shutdown-release-gpu", "shutdown-destroy"):
        command = commands.add_parser(action)
        command.add_argument("instance_id")
        command.add_argument(
            "--confirm",
            action="store_true",
            help="Required acknowledgement for this state-changing action.",
        )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        token = _token_from_environment()
        with XiangongyunClient(token, base_url=args.api_base) as client:
            if args.command == "balance":
                _print_json({"balance": client.balance()})
            elif args.command == "instances":
                _print_json({"instances": client.instances()})
            elif args.command == "instance":
                _print_json(client.instance(args.instance_id))
            elif args.command == "deploy-flashhead":
                if not args.confirm_billing:
                    parser.error("deploy-flashhead requires --confirm-billing")
                deployment = FlashHeadDeployment(
                    gpu_model=args.gpu_model,
                    gpu_count=args.gpu_count,
                    data_center_id=args.data_center_id,
                    image=args.image,
                    image_type=args.image_type,
                    name=args.name,
                )
                _print_json({"id": client.deploy(deployment), "billing_started": True})
            elif args.command == "wait":
                _print_json(
                    client.wait_for_status(
                        args.instance_id,
                        status=args.status,
                        timeout_seconds=args.timeout,
                    )
                )
            else:
                if not args.confirm:
                    parser.error(f"{args.command} requires --confirm")
                api_action = args.command.replace("-", "_")
                _print_json(
                    {
                        "id": args.instance_id,
                        "action": api_action,
                        "result": client.action(args.instance_id, api_action),
                    }
                )
    except XiangongyunError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
