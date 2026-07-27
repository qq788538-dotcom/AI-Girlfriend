#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

TASK_RE = re.compile(r"Task (?P<task>\S+) Pre-processing Report:")
STAGE_RE = re.compile(r"^\s*-\s+(?P<stage>[a-z_]+)\s*:\s*(?P<seconds>[0-9.]+)s")
CHUNK_RE = re.compile(
    r"生成完成 (?P<index>\d+)/(?P<total>\d+), "
    r"frames=(?P<frames>\d+), 一个chunk耗时:(?P<seconds>[0-9.]+)s"
)


def parse_tasks(log: str) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line in log.splitlines():
        if match := TASK_RE.search(line):
            current = {
                "task_id": match.group("task"),
                "preprocessing_s": {},
                "chunks": [],
            }
            tasks.append(current)
            continue
        if current is None:
            continue
        if match := STAGE_RE.match(line):
            preprocessing = current["preprocessing_s"]
            assert isinstance(preprocessing, dict)
            preprocessing[match.group("stage")] = float(match.group("seconds"))
            continue
        if match := CHUNK_RE.search(line):
            chunks = current["chunks"]
            assert isinstance(chunks, list)
            frames = int(match.group("frames"))
            seconds = float(match.group("seconds"))
            chunks.append(
                {
                    "index": int(match.group("index")),
                    "total": int(match.group("total")),
                    "frames": frames,
                    "seconds": seconds,
                    "throughput_fps": frames / seconds,
                }
            )
    return tasks


def summarize(tasks: list[dict[str, object]]) -> dict[str, object]:
    prompt_times: list[float] = []
    first_chunks: list[dict[str, float]] = []
    steady_chunks: list[dict[str, float]] = []
    all_chunks: list[dict[str, float]] = []
    prompt_plus_first: list[float] = []

    for task in tasks:
        preprocessing = task["preprocessing_s"]
        chunks = task["chunks"]
        assert isinstance(preprocessing, dict)
        assert isinstance(chunks, list)
        prompt = preprocessing.get("prompt_init")
        if isinstance(prompt, float):
            prompt_times.append(prompt)
        for chunk in chunks:
            assert isinstance(chunk, dict)
            item = {
                "frames": float(chunk["frames"]),
                "seconds": float(chunk["seconds"]),
            }
            all_chunks.append(item)
            if int(chunk["index"]) == 1:
                first_chunks.append(item)
                if isinstance(prompt, float):
                    prompt_plus_first.append(prompt + item["seconds"])
            else:
                steady_chunks.append(item)

    def throughput(chunks: list[dict[str, float]]) -> float | None:
        seconds = sum(item["seconds"] for item in chunks)
        if not seconds:
            return None
        return sum(item["frames"] for item in chunks) / seconds

    return {
        "tasks": len(tasks),
        "chunks": len(all_chunks),
        "prompt_init_mean_s": mean(prompt_times) if prompt_times else None,
        "first_chunk_throughput_fps": throughput(first_chunks),
        "steady_chunk_throughput_fps": throughput(steady_chunks),
        "all_chunk_throughput_fps": throughput(all_chunks),
        "prompt_plus_first_chunk_mean_s": (
            mean(prompt_plus_first) if prompt_plus_first else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize SoulX-LiveAct preprocessing and chunk throughput logs."
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("--last-tasks", type=int, default=30)
    args = parser.parse_args()
    if not args.log.is_file():
        parser.error(f"log does not exist: {args.log}")
    if args.last_tasks <= 0:
        parser.error("--last-tasks must be positive")

    tasks = parse_tasks(args.log.read_text(encoding="utf-8", errors="replace"))
    selected = tasks[-args.last_tasks :]
    print(
        json.dumps(
            {
                "source": str(args.log),
                "selection": {"last_tasks": args.last_tasks},
                "summary": summarize(selected),
                "tasks": selected,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
