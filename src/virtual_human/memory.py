from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OpenVikingMemory:
    """Fail-open client for the virtual girlfriend's isolated memory service."""

    _SESSION_MEMORY_POLICY = {
        "self": {"enabled": True},
        "peer": {"enabled": False},
        "memory_types": ["entities", "preferences", "events"],
        "working_memory": {"enabled": False},
    }

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        timeout_seconds: float = 3.0,
        search_limit: int = 4,
        score_threshold: float = 0.01,
        max_context_chars: int = 1800,
        agent_id: str = "virtual-girlfriend",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self.agent_id = agent_id
        self.search_limit = search_limit
        self.score_threshold = score_threshold
        self.max_context_chars = max_context_chars
        self._initialized_sessions: set[str] = set()
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    async def recall(self, query: str) -> str:
        """Return a compact prompt fragment containing only relevant memories."""
        try:
            response = await self._client.post(
                "api/v1/search/recall",
                json={
                    "query": query,
                    "quotas": {
                        "events": self.search_limit,
                        "entities": self.search_limit,
                        "preferences": self.search_limit,
                        "experiences": 0,
                    },
                    # Ask OpenViking for enough candidates from every memory
                    # type, then apply the tighter prompt budget after
                    # interleaving the categories below.
                    "max_chars": min(self.max_context_chars * 3, 10_000),
                    "min_score": self.score_threshold,
                    "render": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as error:
            logger.warning("OpenViking recall unavailable; continuing without memory: %s", error)
            return ""

        result = payload.get("result") if isinstance(payload, dict) else None
        memories = result.get("entries") if isinstance(result, dict) else None
        if not isinstance(memories, list):
            return ""

        buckets: dict[str, list[dict[str, Any]]] = {
            "entities": [],
            "preferences": [],
            "events": [],
            "experiences": [],
            "other": [],
        }
        for item in memories:
            if not isinstance(item, dict):
                continue
            memory_type = str(item.get("type") or "")
            buckets.get(memory_type, buckets["other"]).append(item)
        ordered_memories: list[dict[str, Any]] = []
        largest_bucket = max((len(bucket) for bucket in buckets.values()), default=0)
        for index in range(largest_bucket):
            for memory_type in ("entities", "preferences", "events", "experiences", "other"):
                bucket = buckets[memory_type]
                if index < len(bucket):
                    ordered_memories.append(bucket[index])

        selected: list[tuple[str, str]] = []
        used = 0
        type_labels = {
            "entities": "长期实体",
            "preferences": "稳定偏好",
            "events": "历史事件",
            "experiences": "经验",
        }
        for item in ordered_memories:
            text = str(
                item.get("content")
                or item.get("summary")
                or item.get("abstract")
                or item.get("overview")
                or ""
            ).strip()
            if not text:
                continue
            remaining = self.max_context_chars - used
            if remaining <= 0:
                break
            memory_type = str(item.get("type") or "")
            uri = str(item.get("uri") or "")
            is_user_profile = memory_type == "entities" and (
                "/entities/用户/" in uri
                or "/entities/user/" in uri.lower()
                or uri.endswith("/profile.md")
            )
            selected_type = "user_profile" if is_user_profile else memory_type
            label = (
                "用户资料·最高优先"
                if is_user_profile
                else type_labels.get(memory_type, "其他")
            )
            prefix = f"- [{label}] "
            clipped = text[: max(0, remaining - len(prefix))]
            if not clipped:
                break
            selected.append((selected_type or "other", f"{prefix}{clipped}"))
            used += len(prefix) + len(clipped)

        # Put historical evidence first and the highest-scoring structured
        # facts last. Models pay more attention to the tail of a long system
        # prompt, so an old assistant guess must not override a user entity.
        ordered_lines: list[str] = []
        for memory_type in (
            "events",
            "experiences",
            "other",
            "preferences",
            "entities",
            "user_profile",
        ):
            matching = [line for item_type, line in selected if item_type == memory_type]
            if memory_type in {"preferences", "entities", "user_profile"}:
                matching.reverse()
            ordered_lines.extend(matching)
        return "\n".join(ordered_lines)

    async def record_turn(self, session_id: str, user_text: str, assistant_text: str) -> bool:
        try:
            if session_id not in self._initialized_sessions:
                create_response = await self._client.post(
                    "api/v1/sessions",
                    json={
                        "session_id": session_id,
                        "memory_policy": self._SESSION_MEMORY_POLICY,
                    },
                )
                if create_response.status_code != 409:
                    create_response.raise_for_status()
                self._initialized_sessions.add(session_id)
            response = await self._client.post(
                f"api/v1/sessions/{session_id}/messages/batch",
                json={
                    "messages": [
                        {"role": "user", "content": user_text},
                        {
                            "role": "assistant",
                            "content": assistant_text,
                            "agent_id": self.agent_id,
                        },
                    ]
                },
            )
            response.raise_for_status()
            return True
        except httpx.HTTPError as error:
            logger.warning("OpenViking turn capture failed; continuing without memory: %s", error)
            return False

    async def commit(self, session_id: str) -> bool:
        try:
            response = await self._client.post(
                f"api/v1/sessions/{session_id}/commit",
                json={"keep_recent_count": 0},
            )
            response.raise_for_status()
            return True
        except httpx.HTTPError as error:
            logger.warning("OpenViking commit failed; conversation remains available: %s", error)
            return False

    async def health(self) -> dict[str, Any]:
        response = await self._client.get("health")
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {"status": "unknown"}

    async def close(self) -> None:
        await self._client.aclose()
