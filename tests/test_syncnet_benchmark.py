from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/benchmark-syncnet.py"
SPEC = importlib.util.spec_from_file_location("benchmark_syncnet", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_syncnet_gate_accepts_current_realtime_target() -> None:
    assert (
        MODULE.metric_violations(
            offset_frames=-2,
            confidence=6.4,
            max_abs_offset_frames=2,
            min_confidence=4.0,
        )
        == []
    )


def test_syncnet_gate_rejects_offset_and_low_confidence() -> None:
    violations = MODULE.metric_violations(
        offset_frames=4,
        confidence=2.5,
        max_abs_offset_frames=2,
        min_confidence=4.0,
    )

    assert len(violations) == 2
    assert "offset" in violations[0]
    assert "confidence" in violations[1]
