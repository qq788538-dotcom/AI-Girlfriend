from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze-liveact-log.py"
SPEC = importlib.util.spec_from_file_location("analyze_liveact_log", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_liveact_log_parser_reports_prompt_and_chunk_throughput() -> None:
    tasks = MODULE.parse_tasks(
        """
Task sample-a Pre-processing Report:
 - audio_proc          : 0.1000s
 - prompt_init         : 12.0000s
生成完成 1/2, frames=21, 一个chunk耗时:4.2000s
生成完成 2/2, frames=32, 一个chunk耗时:6.4000s
Task sample-b Pre-processing Report:
 - prompt_init         : 10.0000s
生成完成 1/1, frames=21, 一个chunk耗时:3.5000s
"""
    )

    summary = MODULE.summarize(tasks)

    assert summary["tasks"] == 2
    assert summary["chunks"] == 3
    assert summary["prompt_init_mean_s"] == 11
    assert summary["first_chunk_throughput_fps"] == 21 / 3.85
    assert summary["steady_chunk_throughput_fps"] == 5
    assert summary["prompt_plus_first_chunk_mean_s"] == 14.85
