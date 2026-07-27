#!/usr/bin/env python3
"""Run the pinned SoulX-LiveAct demo with its HTTP listener forced to loopback."""

from __future__ import annotations

import os
import runpy
import sys
import warnings
from collections import OrderedDict
from importlib import import_module
from pathlib import Path

from flask import Flask


def _install_sdpa_fallback() -> None:
    """Route LiveAct's direct FlashAttention calls through PyTorch SDPA."""
    if os.environ.get("VH_LIVEACT_FORCE_SDPA", "0") != "1":
        return

    # LiveAct catches ModuleNotFoundError but not binary-loader ImportError.
    # Mask both optional packages before importing its attention module so a
    # stale/incompatible wheel cannot prevent the supported SDPA fallback.
    sys.modules["flash_attn"] = None
    sys.modules["flash_attn_interface"] = None

    import torch
    import torch.nn.functional as functional

    attention_module = import_module("wan.modules.attention")

    def sdpa_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens: object = None,
        k_lens: object = None,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        q_scale: float | None = None,
        causal: bool = False,
        window_size: tuple[int, int] = (-1, -1),
        deterministic: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        version: int | None = None,
    ) -> torch.Tensor:
        del deterministic, version
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                "LiveAct SDPA fallback ignores padding lengths, matching its "
                "upstream fallback implementation.",
                stacklevel=2,
            )
        if window_size != (-1, -1):
            warnings.warn(
                "LiveAct SDPA fallback does not implement sliding-window attention.",
                stacklevel=2,
            )

        out_dtype = q.dtype
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)
        if q_scale is not None:
            q = q * q_scale

        kwargs: dict[str, object] = {
            "attn_mask": None,
            "is_causal": causal,
            "dropout_p": dropout_p,
        }
        if softmax_scale is not None:
            kwargs["scale"] = softmax_scale
        if q.size(1) != k.size(1):
            kwargs["enable_gqa"] = True

        out = functional.scaled_dot_product_attention(q, k, v, **kwargs)
        return out.transpose(1, 2).contiguous().to(out_dtype)

    attention_module.flash_attention = sdpa_attention
    modules_package = sys.modules.get("wan.modules")
    if modules_package is not None:
        modules_package.flash_attention = sdpa_attention
    for module_name in ("wan.modules.clip", "wan.modules.model"):
        loaded_module = sys.modules.get(module_name)
        if loaded_module is not None:
            loaded_module.flash_attention = sdpa_attention
    warnings.warn(
        "SoulX-LiveAct is using the PyTorch SDPA compatibility backend.",
        stacklevel=2,
    )


def _install_t5_cache() -> None:
    """Cache repeated UMT5 outputs without modifying the pinned vendor tree."""
    if os.environ.get("VH_LIVEACT_CACHE_T5", "0") != "1":
        return

    t5_module = import_module("wan.modules.t5")
    encoder_class = t5_module.T5EncoderModel
    original_call = encoder_class.__call__
    max_entries = max(1, int(os.environ.get("VH_LIVEACT_T5_CACHE_ENTRIES", "32")))

    def cached_call(self: object, texts: object, device: object) -> object:
        if isinstance(texts, str):
            text_key: tuple[str, ...] | None = ("str", texts)
        elif isinstance(texts, (list, tuple)) and all(
            isinstance(text, str) for text in texts
        ):
            text_key = ("sequence", *texts)
        else:
            text_key = None

        if text_key is None:
            return original_call(self, texts, device)

        cache = getattr(self, "_vh_t5_cache", None)
        if cache is None:
            cache = OrderedDict()
            setattr(self, "_vh_t5_cache", cache)
        key = (text_key, str(device))
        if key in cache:
            cache.move_to_end(key)
            return cache[key]

        result = original_call(self, texts, device)
        cache[key] = result
        cache.move_to_end(key)
        while len(cache) > max_entries:
            cache.popitem(last=False)
        return result

    encoder_class.__call__ = cached_call
    warnings.warn(
        f"SoulX-LiveAct UMT5 cache enabled ({max_entries} entries).",
        stacklevel=2,
    )


def _install_vae_compile_policy() -> None:
    """Control LightVAE compilation independently from the denoising model."""
    mode = os.environ.get("VH_LIVEACT_VAE_COMPILE_MODE", "static").strip().lower()
    if mode not in {"static", "dynamic", "off"}:
        raise SystemExit(
            "VH_LIVEACT_VAE_COMPILE_MODE must be static, dynamic, or off"
        )
    if mode == "static":
        return

    import torch

    original_compile = torch.compile

    def compile_with_vae_policy(model: object, *args: object, **kwargs: object):
        qualified_name = str(
            getattr(model, "__qualname__", getattr(model, "__name__", ""))
        )
        is_lightvae_decode = qualified_name.endswith("WanVAE.decode")
        if not is_lightvae_decode:
            return original_compile(model, *args, **kwargs)
        if mode == "off":
            warnings.warn(
                "SoulX-LiveAct LightVAE decode torch.compile disabled.",
                stacklevel=2,
            )
            return model
        kwargs["dynamic"] = True
        warnings.warn(
            "SoulX-LiveAct LightVAE decode uses a dynamic torch.compile graph.",
            stacklevel=2,
        )
        return original_compile(model, *args, **kwargs)

    torch.compile = compile_with_vae_policy


def main() -> None:
    project_dir = Path(
        os.environ.get("VH_AUTODL_PROJECT_DIR", "/root/AI-Girlfriend")
    ).resolve()
    liveact_dir = project_dir / "vendor" / "SoulX-LiveAct"
    demo_path = liveact_dir / "demo.py"
    if not demo_path.is_file():
        raise SystemExit(f"LiveAct demo is missing: {demo_path}")

    original_run = Flask.run

    def run_on_loopback(self: Flask, *args: object, **kwargs: object) -> object:
        kwargs["host"] = "127.0.0.1"
        return original_run(self, *args, **kwargs)

    Flask.run = run_on_loopback
    os.chdir(liveact_dir)
    sys.path.insert(0, str(liveact_dir))
    _install_sdpa_fallback()
    _install_t5_cache()
    _install_vae_compile_policy()
    runpy.run_path(str(demo_path), run_name="__main__")


if __name__ == "__main__":
    main()
