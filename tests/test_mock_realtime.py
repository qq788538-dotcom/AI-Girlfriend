import json

from virtual_human.mock_realtime import MockRealtimeSession


async def test_mock_realtime_streams_audio_and_completion() -> None:
    session = MockRealtimeSession(sample_rate=24000)
    await session.start()
    events = session.events()

    created = json.loads(await anext(events))
    assert created["type"] == "session.created"

    await session.send(json.dumps({"type": "response.create"}))

    event_types = []
    while "response.done" not in event_types:
        event = json.loads(await anext(events))
        event_types.append(event["type"])

    assert "response.created" in event_types
    assert "response.output_audio.delta" in event_types
    assert "response.output_audio_transcript.done" in event_types
    assert event_types[-1] == "response.done"
    await session.close()
