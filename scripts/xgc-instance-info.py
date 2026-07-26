#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path


def env_value(path: Path, name: str) -> str:
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            return value.strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read sanitized Xiangongyun instance metadata."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="List every instance instead of the configured deployment only.",
    )
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    token = env_value(project_dir / ".env", "XGC_API_TOKEN")
    instance_id = json.loads(
        (project_dir / "runtime/credentials/xgc-instance.json").read_text()
    )["id"]
    if not token:
        raise SystemExit("XGC_API_TOKEN is not configured")

    url = (
        "https://api.xiangongyun.com/open/instances"
        if args.all
        else f"https://api.xiangongyun.com/open/instance/{instance_id}"
    )
    header_candidates = (
        {"Authorization": f"Bearer {token}"},
        {"Authorization": token},
        {"X-API-Key": token},
    )
    for headers in header_candidates:
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            if error.code in {401, 403}:
                continue
            raise
        data = payload.get("data", payload)
        if args.all:
            if isinstance(data, dict):
                data = data.get("list", [])
            if not isinstance(data, list):
                raise SystemExit("Unexpected XGC instance-list response")
            safe_instances = [
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "status": item.get("status"),
                    "progress": item.get("progress"),
                    "gpu_model": item.get("gpu_model"),
                    "gpu_used": item.get("gpu_used"),
                    "image_id": item.get("image_id"),
                    "image_type": item.get("image_type"),
                    "ssh_domain": item.get("ssh_domain"),
                    "ssh_port": item.get("ssh_port"),
                    "web_url": item.get("web_url"),
                }
                for item in data
                if isinstance(item, dict)
            ]
            print(
                json.dumps(
                    {"instances": safe_instances},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        if isinstance(data, dict) and isinstance(data.get("instance"), dict):
            data = data["instance"]
        if not isinstance(data, dict):
            raise SystemExit("Unexpected XGC instance response")
        safe = {
            "id": data.get("id"),
            "status": data.get("status"),
            "gpu_model": data.get("gpu_model"),
            "gpu_used": data.get("gpu_used"),
            "ssh_domain": data.get("ssh_domain"),
            "ssh_port": data.get("ssh_port"),
            "web_url": data.get("web_url"),
        }
        print(json.dumps(safe, ensure_ascii=False, indent=2))
        return
    raise SystemExit("XGC API authentication failed")


if __name__ == "__main__":
    main()
