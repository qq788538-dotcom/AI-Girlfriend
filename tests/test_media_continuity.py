from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/benchmark-media-continuity.py"
SPEC = importlib.util.spec_from_file_location("benchmark_media_continuity", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_metadata_values_extracts_only_requested_signalstat() -> None:
    log = """
frame:52 pts_time:2.166667
lavfi.signalstats.YAVG=77.1083
lavfi.signalstats.YDIF=1.41157
frame:53 pts_time:2.208333
lavfi.signalstats.YDIF=26.2097
lavfi.signalstats.UDIF=8.5
"""

    assert MODULE.metadata_values(log, "lavfi.signalstats.YDIF") == [
        1.41157,
        26.2097,
    ]


def test_metadata_values_accepts_signed_and_integer_values() -> None:
    log = """
lavfi.signalstats.YDIF=0
lavfi.signalstats.YDIF=-0.25
lavfi.signalstats.YDIF=+3.5
"""

    assert MODULE.metadata_values(log, "lavfi.signalstats.YDIF") == [
        0.0,
        -0.25,
        3.5,
    ]


def test_max_sustained_drop_detects_persistent_latent_collapse() -> None:
    values = [0.78] * 24 + [0.77, 0.76] + [0.64] * 12

    drop, start = MODULE.max_sustained_drop(
        values,
        baseline_frames=24,
        window_frames=8,
    )

    assert round(drop, 3) == 0.14
    assert start == 26


def test_max_sustained_drop_ignores_single_dynamic_frame() -> None:
    values = [0.91] * 24 + [0.70] + [0.90] * 12

    drop, start = MODULE.max_sustained_drop(
        values,
        baseline_frames=24,
        window_frames=8,
    )

    assert round(drop, 3) == 0.035
    assert start == 24
