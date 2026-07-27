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
    # The official model imports SageAttention independently from Wan's
    # attention wrapper. Mask it too so this flag remains a complete rollback
    # path after a SageAttention wheel is installed.
    sys.modules["sageattention"] = None

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


def _patch_reference_cache(source: str) -> str:
    """Cache immutable reference conditioning in the engine process."""
    import_anchor = "import argparse\n"
    warmup_start = "                # 1. 准备假图像\n"
    warmup_end = "                # 2. CLIP\n"
    warmup_store_anchor = (
        "                y = torch.concat([msk, y], dim=1)\n"
        "\n"
        "                # 5. prompt\n"
    )
    generate_start = "            # 3. 图像 / 条件\n"
    generate_end = (
        "            if self.rank == 0:\n"
        "                start_time = time.perf_counter()\n"
        "\n"
        "            edit_prompts = {}\n"
    )
    for label, anchor in (
        ("import", import_anchor),
        ("warmup start", warmup_start),
        ("warmup end", warmup_end),
        ("warmup store", warmup_store_anchor),
        ("generate start", generate_start),
        ("generate end", generate_end),
    ):
        if source.count(anchor) != 1:
            raise SystemExit(
                f"LiveAct reference cache could not find the {label} anchor"
            )

    source = source.replace(
        import_anchor,
        import_anchor + "import hashlib\nfrom collections import OrderedDict\n",
        1,
    )
    warmup_block = (
        "                # 1. 准备生产参考图像（可选）\n"
        "                warmup_reference_path = os.environ.get(\n"
        '                    "VH_LIVEACT_WARMUP_REFERENCE", ""\n'
        "                )\n"
        "                warmup_reference_key = None\n"
        "                if (\n"
        '                    os.environ.get("VH_LIVEACT_CACHE_REFERENCE", "0") == "1"\n'
        "                    and os.path.isfile(warmup_reference_path)\n"
        "                ):\n"
        "                    reference_hasher = hashlib.sha256()\n"
        "                    with open(warmup_reference_path, \"rb\") as reference_file:\n"
        "                        for reference_block in iter(\n"
        '                            lambda: reference_file.read(1024 * 1024), b""\n'
        "                        ):\n"
        "                            reference_hasher.update(reference_block)\n"
        "                    warmup_reference_key = (\n"
        "                        reference_hasher.hexdigest(),\n"
        "                        self.height,\n"
        "                        self.width,\n"
        "                        str(self.device),\n"
        "                    )\n"
        "                    warmup_image = Image.open(warmup_reference_path).convert(\"RGB\")\n"
        "                    cond_image = self.transform(warmup_image).unsqueeze(1).unsqueeze(0).to(\n"
        "                        self.device, torch.bfloat16\n"
        "                    )\n"
        "                else:\n"
        "                    cond_image = torch.randn(\n"
        "                        1, 3, 1, self.height, self.width,\n"
        "                        device=self.device, dtype=torch.bfloat16\n"
        "                    ).clamp_(-1, 1)\n"
    )
    warmup_start_index = source.index(warmup_start)
    warmup_end_index = source.index(warmup_end, warmup_start_index)
    source = (
        source[:warmup_start_index]
        + warmup_block
        + source[warmup_end_index:]
    )
    source = source.replace(
        warmup_store_anchor,
        (
            "                y = torch.concat([msk, y], dim=1)\n"
            "                if warmup_reference_key is not None:\n"
            "                    reference_cache = OrderedDict()\n"
            "                    reference_cache[warmup_reference_key] = (\n"
            "                        cond_image, clip_context, ref_target_masks, y\n"
            "                    )\n"
            "                    self._vh_reference_cache = reference_cache\n"
            "                    if self.rank == 0:\n"
            "                        print(\n"
            '                            "[ReferenceCache] production reference prewarmed",\n'
            "                            flush=True,\n"
            "                        )\n"
            "\n"
            "                # 5. prompt\n"
        ),
        1,
    )

    generate_block = (
        "            # 3. 图像 / 条件\n"
        "            reference_cache_key = None\n"
        "            cached_reference = None\n"
        "            if self.rank == 0:\n"
        "                start_time = time.perf_counter()\n"
        '            if os.environ.get("VH_LIVEACT_CACHE_REFERENCE", "0") == "1":\n'
        "                reference_hasher = hashlib.sha256()\n"
        "                with open(img_path, \"rb\") as reference_file:\n"
        "                    for reference_block in iter(\n"
        '                        lambda: reference_file.read(1024 * 1024), b""\n'
        "                    ):\n"
        "                        reference_hasher.update(reference_block)\n"
        "                reference_cache_key = (\n"
        "                    reference_hasher.hexdigest(),\n"
        "                    self.height,\n"
        "                    self.width,\n"
        "                    str(self.device),\n"
        "                )\n"
        "                reference_cache = getattr(self, \"_vh_reference_cache\", None)\n"
        "                if reference_cache is not None:\n"
        "                    cached_reference = reference_cache.get(reference_cache_key)\n"
        "                    if cached_reference is not None:\n"
        "                        reference_cache.move_to_end(reference_cache_key)\n"
        "\n"
        "            frame_num_init = (sum(self.blksz_lst) - 1) * 4 + 1\n"
        "            torch.manual_seed(self.args.seed)\n"
        "            if cached_reference is not None:\n"
        "                cond_image, clip_context, ref_target_masks, y = cached_reference\n"
        "                if self.rank == 0:\n"
        "                    stats['image_proc'] = time.perf_counter() - start_time\n"
        "                    stats['clip_proc'] = 0.0\n"
        "                    stats['init_y'] = 0.0\n"
        '                    print("[ReferenceCache] hit", flush=True)\n'
        "            else:\n"
        "                image = Image.open(img_path).convert(\"RGB\")\n"
        "                cond_image = self.transform(image).unsqueeze(1).unsqueeze(0).to(\n"
        "                    self.device, torch.bfloat16\n"
        "                )\n"
        "                if self.rank == 0:\n"
        "                    stats['image_proc'] = time.perf_counter() - start_time\n"
        "                    start_time = time.perf_counter()\n"
        "                with torch.no_grad():\n"
        "                    clip_context = self.clip.visual(cond_image)\n"
        "                if self.rank == 0:\n"
        "                    stats['clip_proc'] = time.perf_counter() - start_time\n"
        "                    start_time = time.perf_counter()\n"
        "                ref_target_masks = torch.ones(\n"
        "                    3,\n"
        "                    self.height // self.vae_stride[1],\n"
        "                    self.width // self.vae_stride[2],\n"
        "                    device=self.device,\n"
        "                    dtype=torch.bfloat16\n"
        "                )\n"
        "                msk = get_msk(frame_num_init, cond_image, self.vae_stride, self.device)\n"
        "                video_frames_placeholder = torch.zeros(\n"
        "                    1,\n"
        "                    cond_image.shape[1],\n"
        "                    frame_num_init - cond_image.shape[2],\n"
        "                    self.height,\n"
        "                    self.width,\n"
        "                    device=self.device,\n"
        "                    dtype=torch.bfloat16\n"
        "                )\n"
        "                padding_frames = torch.concat(\n"
        "                    [cond_image, video_frames_placeholder], dim=2\n"
        "                )\n"
        "                y = self.vae.encode(padding_frames).to(self.device).unsqueeze(0)\n"
        "                y = torch.concat([msk, y], dim=1)\n"
        "                if self.rank == 0:\n"
        "                    stats['init_y'] = time.perf_counter() - start_time\n"
        "                if reference_cache_key is not None:\n"
        "                    reference_cache = getattr(self, \"_vh_reference_cache\", None)\n"
        "                    if reference_cache is None:\n"
        "                        reference_cache = OrderedDict()\n"
        "                        self._vh_reference_cache = reference_cache\n"
        "                    reference_cache[reference_cache_key] = (\n"
        "                        cond_image, clip_context, ref_target_masks, y\n"
        "                    )\n"
        "                    reference_cache.move_to_end(reference_cache_key)\n"
        "                    max_reference_entries = max(\n"
        "                        1,\n"
        "                        int(os.environ.get(\n"
        '                            "VH_LIVEACT_REFERENCE_CACHE_ENTRIES", "4"\n'
        "                        )),\n"
        "                    )\n"
        "                    while len(reference_cache) > max_reference_entries:\n"
        "                        reference_cache.popitem(last=False)\n"
        "                    if self.rank == 0:\n"
        '                        print("[ReferenceCache] miss; cached", flush=True)\n'
        "\n"
    )
    generate_start_index = source.index(generate_start)
    generate_end_index = source.index(generate_end, generate_start_index)
    return (
        source[:generate_start_index]
        + generate_block
        + source[generate_end_index:]
    )


def _prepare_demo_path(demo_path: Path) -> Path:
    """Build a same-directory runtime copy that preserves the final audio frames."""
    fix_tail_frames = os.environ.get("VH_LIVEACT_FIX_TAIL_FRAMES", "0") == "1"
    cache_reference = os.environ.get("VH_LIVEACT_CACHE_REFERENCE", "0") == "1"
    if not fix_tail_frames and not cache_reference:
        return demo_path

    source = demo_path.read_text(encoding="utf-8")
    if cache_reference:
        source = _patch_reference_cache(source)
    import_anchor = "import argparse\n"
    count_anchor = (
        "            iter_total_num = int(audio_len_sec / "
        "(self.vae_stride[0] * self.blksz_lst[-1] / fps)) + 1\n"
        "            pre_latent = None\n"
    )
    write_anchor = (
        "                    chunk_bytes, num_frames_this_chunk = "
        "tensor_chunk_to_rgb_bytes(_videos)\n"
    )
    warmup_anchor = '                        texts="A person is speaking naturally.",\n'
    if fix_tail_frames:
        if source.count(import_anchor) != 1:
            raise SystemExit("LiveAct tail fix could not find the import anchor")
        if source.count(count_anchor) != 1:
            raise SystemExit("LiveAct tail fix could not find the chunk-count anchor")
        if source.count(write_anchor) != 1:
            raise SystemExit("LiveAct tail fix could not find the chunk-write anchor")
        if source.count(warmup_anchor) != 1:
            raise SystemExit("LiveAct runtime patch could not find the warmup prompt")

        source = source.replace(import_anchor, import_anchor + "import math\n", 1)
        source = source.replace(
            count_anchor,
            (
                "            first_chunk_frames = "
                "(self.blksz_lst[0] - 1) * self.vae_stride[0] + 1\n"
                "            steady_chunk_frames = "
                "self.blksz_lst[-1] * self.vae_stride[0]\n"
                "            target_total_frames = math.ceil(audio_len_sec * fps)\n"
                "            remaining_frames = max(0, target_total_frames - "
                "first_chunk_frames)\n"
                "            iter_total_num = 1 + math.ceil("
                "remaining_frames / steady_chunk_frames)\n"
                "            generated_frames = 0\n"
                "            pre_latent = None\n"
            ),
            1,
        )
        source = source.replace(
            write_anchor,
            (
                "                    frames_remaining = "
                "target_total_frames - generated_frames\n"
                "                    _videos = _videos[:, :, :frames_remaining]\n"
                + write_anchor
                + "                    generated_frames += num_frames_this_chunk\n"
            ),
            1,
        )
        source = source.replace(
            warmup_anchor,
            (
                "                        texts=os.environ.get(\n"
                '                            "VH_LIVEACT_WARMUP_PROMPT",\n'
                '                            "A person is speaking naturally.",\n'
                "                        ),\n"
            ),
            1,
        )
    patched_path = demo_path.with_name(".autodl-demo-tailfix.py")
    if not patched_path.is_file() or patched_path.read_text(encoding="utf-8") != source:
        patched_path.write_text(source, encoding="utf-8")
    warnings.warn(
        "SoulX-LiveAct final chunk count and frame trimming fix enabled.",
        stacklevel=2,
    )
    return patched_path


def main() -> None:
    project_dir = Path(
        os.environ.get("VH_AUTODL_PROJECT_DIR", "/root/AI-Girlfriend")
    ).resolve()
    liveact_dir = project_dir / "vendor" / "SoulX-LiveAct"
    demo_path = liveact_dir / "demo.py"
    if not demo_path.is_file():
        raise SystemExit(f"LiveAct demo is missing: {demo_path}")
    demo_path = _prepare_demo_path(demo_path)

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
