from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioChunk:
    response_id: str
    chunk_index: int
    pts_ms: float
    duration_ms: float
    sample_rate: int
    pcm16: bytes


class AudioTimeline:
    """Assigns deterministic audio-master timestamps to streamed PCM16 chunks."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._response_id: str | None = None
        self._sample_offset = 0
        self._chunk_index = 0

    def reset(self, response_id: str | None = None) -> None:
        self._response_id = response_id
        self._sample_offset = 0
        self._chunk_index = 0

    def ingest(self, response_id: str, pcm16: bytes) -> AudioChunk:
        if len(pcm16) % 2:
            raise ValueError("PCM16 payload must contain a whole number of samples")
        if self._response_id != response_id:
            self.reset(response_id)

        sample_count = len(pcm16) // 2
        pts_ms = self._sample_offset * 1000 / self.sample_rate
        duration_ms = sample_count * 1000 / self.sample_rate
        chunk = AudioChunk(
            response_id=response_id,
            chunk_index=self._chunk_index,
            pts_ms=pts_ms,
            duration_ms=duration_ms,
            sample_rate=self.sample_rate,
            pcm16=pcm16,
        )
        self._sample_offset += sample_count
        self._chunk_index += 1
        return chunk

    @property
    def buffered_duration_ms(self) -> float:
        return self._sample_offset * 1000 / self.sample_rate
