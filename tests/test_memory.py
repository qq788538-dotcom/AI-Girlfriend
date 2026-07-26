from __future__ import annotations

import httpx

from virtual_human.memory import OpenVikingMemory


async def test_openviking_memory_recalls_and_records_turns() -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.path.endswith("/search/recall"):
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "result": {
                        "entries": [
                            {
                                "type": "preferences",
                                "abstract": "用户不喜欢太甜的咖啡。",
                                "score": 0.82,
                            }
                        ]
                    },
                },
            )
        if request.url.path == "/api/v1/sessions":
            return httpx.Response(
                200,
                json={"status": "ok", "result": {"session_id": "session-1"}},
            )
        if request.url.path.endswith("/messages/batch"):
            return httpx.Response(200, json={"status": "ok", "result": {"added": 2}})
        if request.url.path.endswith("/commit"):
            return httpx.Response(
                200,
                json={"status": "ok", "result": {"status": "accepted", "task_id": "task-1"}},
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    memory = OpenVikingMemory(
        "http://memory.test",
        transport=httpx.MockTransport(handler),
    )
    recalled = await memory.recall("帮我选杯咖啡")
    assert recalled == "- [稳定偏好] 用户不喜欢太甜的咖啡。"
    assert await memory.record_turn("session-1", "我不想喝甜的。", "那就选无糖拿铁。")
    assert await memory.commit("session-1")
    assert [request.url.path for request in observed] == [
        "/api/v1/search/recall",
        "/api/v1/sessions",
        "/api/v1/sessions/session-1/messages/batch",
        "/api/v1/sessions/session-1/commit",
    ]
    assert observed[1].read() == (
        b'{"session_id":"session-1","memory_policy":{"self":{"enabled":true},'
        b'"peer":{"enabled":false},"memory_types":["entities","preferences","events"],'
        b'"working_memory":{"enabled":false}}}'
    )
    assert observed[-1].read() == b'{"keep_recent_count":0}'
    await memory.close()


async def test_structured_entity_is_placed_after_conflicting_history() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "result": {
                    "entries": [
                        {
                            "type": "events",
                            "content": "助手过去误以为用户没有说过昵称。",
                        },
                        {
                            "type": "entities",
                            "uri": "viking://user/default/memories/entities/用户/小猫.md",
                            "content": "用户明确告知昵称是小猫。",
                        },
                    ]
                },
            },
        )

    memory = OpenVikingMemory(
        "http://memory.test",
        transport=httpx.MockTransport(handler),
    )

    recalled = await memory.recall("我叫什么名字")

    assert recalled.endswith("[用户资料·最高优先] 用户明确告知昵称是小猫。")
    await memory.close()


async def test_openviking_memory_fails_open() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "offline"})

    memory = OpenVikingMemory(
        "http://memory.test",
        transport=httpx.MockTransport(handler),
    )
    assert await memory.recall("你还记得吗") == ""
    assert not await memory.record_turn("session-1", "你好", "你好呀")
    assert not await memory.commit("session-1")
    await memory.close()
