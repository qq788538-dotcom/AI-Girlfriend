#!/usr/bin/env python3
"""Run the pinned SoulX-LiveAct demo with its HTTP listener forced to loopback."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from flask import Flask


def main() -> None:
    project_dir = Path(
        os.environ.get("VH_AUTODL_PROJECT_DIR", "/root/AI-Girlfriend")
    ).resolve()
    liveact_dir = project_dir / "vendor" / "SoulX-LiveAct"
    demo_path = liveact_dir / "demo.py"
    if not demo_path.is_file():
        raise SystemExit(f"LiveAct demo is missing: {demo_path}")

    original_run = Flask.run

    def run_on_loopback(self: Flask, *args: object, **kwargs: object) -> object:
        kwargs["host"] = "127.0.0.1"
        return original_run(self, *args, **kwargs)

    Flask.run = run_on_loopback
    os.chdir(liveact_dir)
    sys.path.insert(0, str(liveact_dir))
    runpy.run_path(str(demo_path), run_name="__main__")


if __name__ == "__main__":
    main()
