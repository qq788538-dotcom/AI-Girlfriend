from __future__ import annotations

import hashlib
from pathlib import Path

# Changing either value is an identity migration and requires the user's
# explicit approval. Benchmarks and renderer experiments must use this same
# character; they must never substitute the source video's character.
LOCKED_AVATAR_NAME = "avatar-ai-girlfriend-v6.png"
LOCKED_AVATAR_SHA256 = "c8afa1d691711330e525db0048dd21e37ab71c46476a7504d0e75b4ae8fcd162"


def avatar_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_locked_avatar(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Locked avatar image is missing: {path}")
    digest = avatar_sha256(path)
    if path.name != LOCKED_AVATAR_NAME or digest != LOCKED_AVATAR_SHA256:
        raise ValueError(
            "Avatar identity is locked. Refusing to replace the approved virtual girlfriend."
        )
    return digest
