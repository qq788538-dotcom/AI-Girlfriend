#!/usr/bin/env python3
"""A/B PyTorch SDPA and SageAttention with SoulX-LiveAct-shaped tensors."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.nn.functional as F

PRESETS = {
    # 416x720 / 16x spatial compression = 26x45 = 1170 tokens per latent frame.
    "first": (7_020, 7_020),
    "steady": (9_360, 16_380),
    "image-cross": (9_360, 257),
    "text-cross": (9_360, 512),
}


@dataclass(frozen=True)
class Timing:
    median_ms: float
    mean_ms: float
    p10_ms: float
    p90_ms: float
    iterations: int


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def benchmark(
    function: Callable[[], torch.Tensor], *, warmup: int, iterations: int
) -> tuple[torch.Tensor, Timing]:
    for _ in range(warmup):
        output = function()
    torch.cuda.synchronize()

    times: list[float] = []
    output = function()
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return output, Timing(
        median_ms=statistics.median(times),
        mean_ms=statistics.mean(times),
        p10_ms=quantile(times, 0.1),
        p90_ms=quantile(times, 0.9),
        iterations=iterations,
    )


def sdpa_nhd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=False,
    )
    return output.transpose(1, 2)


def tensor_quality(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_f = reference.float()
    candidate_f = candidate.float()
    difference = (reference_f - candidate_f).abs()
    cosine = F.cosine_similarity(
        reference_f.reshape(1, -1), candidate_f.reshape(1, -1)
    )
    signal = reference_f.square().mean()
    noise = (reference_f - candidate_f).square().mean()
    return {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(reference_f - candidate_f)
            / torch.linalg.vector_norm(reference_f)
        ),
        "cosine_similarity": float(cosine),
        "snr_db": float(10 * torch.log10(signal / noise)) if noise else math.inf,
        "nan_count": float(torch.isnan(candidate_f).sum()),
        "inf_count": float(torch.isinf(candidate_f).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="steady")
    parser.add_argument("--q-tokens", type=int)
    parser.add_argument("--kv-tokens", type=int)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations must be positive")

    try:
        import sageattention
        from sageattention import sageattn
    except ImportError as error:
        parser.error(f"SageAttention is not importable: {error}")

    preset_q, preset_kv = PRESETS[args.preset]
    q_tokens = args.q_tokens or preset_q
    kv_tokens = args.kv_tokens or preset_kv
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)
    q = torch.randn(
        1, q_tokens, args.heads, args.head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        1, kv_tokens, args.heads, args.head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        1, kv_tokens, args.heads, args.head_dim, device=device, dtype=dtype
    )

    torch.cuda.reset_peak_memory_stats()
    sdpa_output, sdpa_timing = benchmark(
        lambda: sdpa_nhd(q, k, v), warmup=args.warmup, iterations=args.iterations
    )
    sdpa_peak = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    sage_output, sage_timing = benchmark(
        lambda: sageattn(q, k, v, tensor_layout="NHD", is_causal=False),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    sage_peak = torch.cuda.max_memory_allocated()

    result = {
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "sageattention": getattr(sageattention, "__version__", "unknown"),
            "dtype": str(dtype),
        },
        "shape": {
            "preset": args.preset,
            "q": list(q.shape),
            "k": list(k.shape),
            "v": list(v.shape),
            "tensor_layout": "NHD",
            "is_causal": False,
        },
        "sdpa": {
            "timing": asdict(sdpa_timing),
            "peak_allocated_mib": sdpa_peak / 1024**2,
        },
        "sageattention": {
            "timing": asdict(sage_timing),
            "peak_allocated_mib": sage_peak / 1024**2,
        },
        "speedup": sdpa_timing.median_ms / sage_timing.median_ms,
        "quality_vs_sdpa": tensor_quality(sdpa_output, sage_output),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
